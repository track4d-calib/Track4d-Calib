"""Refine image-pair overlap with a second SuperPoint/SuperGlue pass."""
from pathlib import Path
import json
import shutil
import time

import cv2
import h5py
import numpy as np
import torch


def crop_bounds(points, shape, margin=0.1):
    """Expand each side by a fraction of the matched bounding-box extent."""
    low, high = points.min(axis=0), points.max(axis=0)
    padding = (high - low) * margin
    low = np.maximum(0, np.floor(low - padding)).astype(int)
    high = np.minimum([shape[1], shape[0]], np.ceil(high + padding) + 1).astype(int)
    return (*low, *high)


def restore_points(points, bounds, resized_shape):
    x0, y0, x1, y1 = bounds
    height, width = resized_shape
    scale = np.array([(x1 - x0) / width, (y1 - y0) / height])
    return (points + 0.5) * scale - 0.5 + [x0, y0]


def geometric_model(left, right, threshold_px=2.0):
    """Estimate epipolar support, returning no inliers if USAC cannot fit a model."""
    if len(left) < 8:
        return None, np.zeros(len(left), dtype=bool)
    cv2.setRNGSeed(0)
    try:
        matrix, mask = cv2.findFundamentalMat(
            np.asarray(left, np.float32), np.asarray(right, np.float32),
            cv2.USAC_MAGSAC, threshold_px, 0.999, 10000,
        )
    except cv2.error as error:
        if error.code != -215 or error.err != "!model.empty()" or error.func != "setModelParameters":
            raise
        return None, np.zeros(len(left), dtype=bool)
    inliers = np.zeros(len(left), dtype=bool) if mask is None else mask.ravel().astype(bool)
    return matrix, inliers


def epipolar_error(matrix, left, right):
    """Return square-root Sampson error in original-image pixels."""
    x, y = np.r_[left, 1.0], np.r_[right, 1.0]
    fx, fy = matrix @ x, matrix.T @ y
    denominator = np.dot(fx[:2], fx[:2]) + np.dot(fy[:2], fy[:2])
    if denominator <= 1e-12:
        return float('inf')
    return float(abs(y @ fx) / np.sqrt(denominator))


class KeypointIndex:
    """Keep shared image-space feature IDs across independently cropped pairs."""
    def __init__(self, points, scores, descriptors):
        self.points = list(points)
        self.scores = list(scores)
        self.descriptors = list(descriptors.T)
        self.cells = {}
        for index, point in enumerate(points):
            self.cells.setdefault(tuple(np.floor(point).astype(int)), []).append(index)

    def add(self, point, score, descriptor):
        cell = tuple(np.floor(point).astype(int))
        nearby = [index for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                  for index in self.cells.get((cell[0] + dx, cell[1] + dy), [])]
        if nearby:
            distances = np.linalg.norm(np.asarray([self.points[i] for i in nearby]) - point, axis=1)
            nearest = int(np.argmin(distances))
            if distances[nearest] <= 1.0:
                return nearby[nearest]
        index = len(self.points)
        self.points.append(point)
        self.scores.append(score)
        self.descriptors.append(descriptor)
        self.cells.setdefault(cell, []).append(index)
        return index


@torch.inference_mode()
def refine_overlap(image_dir, pairs_path, features_path, matches_path, feature_conf, match_conf,
                   resize_max=1024, margin=0.1, geometry_threshold_px=2.0):
    """Append verified crop matches; retain the first pass and reverify in COLMAP."""
    from hloc import extractors, matchers
    from hloc.utils.base_model import dynamic_load
    from hloc.utils.io import get_matches
    from hloc.utils.parsers import names_to_pair

    started = time.perf_counter()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    extractor = dynamic_load(extractors, feature_conf['model']['name'])(feature_conf['model']).eval().to(device)
    matcher = dynamic_load(matchers, match_conf['model']['name'])(match_conf['model']).eval().to(device)
    pairs = list(dict.fromkeys(tuple(line.split()) for line in Path(pairs_path).read_text().splitlines() if line.strip()))
    names = sorted({name for pair in pairs for name in pair})
    images, indices = {}, {}
    with h5py.File(features_path, 'r') as features:
        for name in names:
            images[name] = cv2.imread(str(Path(image_dir) / name), cv2.IMREAD_GRAYSCALE)
            if images[name] is None:
                raise FileNotFoundError(name)
            group = features[name]
            indices[name] = KeypointIndex(group['keypoints'][:], group['scores'][:], group['descriptors'][:])

    def extract(image, bounds):
        x0, y0, x1, y1 = bounds
        crop = image[y0:y1, x0:x1]
        scale = resize_max / max(crop.shape)
        width, height = max(1, round(crop.shape[1] * scale)), max(1, round(crop.shape[0] * scale))
        crop = cv2.resize(crop, (width, height), interpolation=cv2.INTER_LINEAR)
        tensor = torch.from_numpy(crop.astype(np.float32) / 255.0)[None, None].to(device)
        prediction = extractor({'image': tensor})
        data = {key: prediction[key][0][None] for key in ('keypoints', 'scores', 'descriptors')}
        data['image'] = tensor
        original = restore_points(data['keypoints'][0].cpu().numpy(), bounds, crop.shape)
        return data, original

    combined, rows = [], []
    for name0, name1 in pairs:
        matches, scores = get_matches(matches_path, name0, name1)
        idx0, idx1 = indices[name0], indices[name1]
        first0, first1 = np.asarray(idx0.points)[matches[:, 0]], np.asarray(idx1.points)[matches[:, 1]]
        _, valid = geometric_model(first0, first1)
        row = {'pair': [name0, name1], 'first_matches': len(matches), 'first_inliers': int(valid.sum()), 'added': 0, 'merged_geometry_rejected': 0}
        merged = [tuple(map(int, match)) for match in matches]
        confidence = list(map(float, scores))
        if valid.sum() >= 16:
            bounds0 = crop_bounds(first0[valid], images[name0].shape, margin)
            bounds1 = crop_bounds(first1[valid], images[name1].shape, margin)
            row['bounds'] = [list(map(int, bounds0)), list(map(int, bounds1))]
            if all(min(b[2] - b[0], b[3] - b[1]) >= 32 for b in (bounds0, bounds1)):
                data0, points0 = extract(images[name0], bounds0)
                data1, points1 = extract(images[name1], bounds1)
                prediction = matcher({**{k+'0': v for k, v in data0.items()}, **{k+'1': v for k, v in data1.items()}})
                mapping = prediction['matches0'][0].cpu().numpy()
                source = np.flatnonzero(mapping >= 0)
                target = mapping[source]
                matrix, keep = geometric_model(points0[source], points1[target], geometry_threshold_px)
                row['crop_matches'], row['crop_inliers'] = len(source), int(keep.sum())
                used0, used1 = set(matches[:, 0]), set(matches[:, 1])
                detections0 = data0['scores'][0].cpu().numpy()
                detections1 = data1['scores'][0].cpu().numpy()
                descriptors0 = data0['descriptors'][0].cpu().numpy()
                descriptors1 = data1['descriptors'][0].cpu().numpy()
                crop_scores = prediction['matching_scores0'][0].cpu().numpy()
                for a, b in zip(source[keep], target[keep]):
                    i = idx0.add(points0[a], detections0[a], descriptors0[:, a])
                    j = idx1.add(points1[b], detections1[b], descriptors1[:, b])
                    if i not in used0 and j not in used1:
                        if epipolar_error(matrix, idx0.points[i], idx1.points[j]) > geometry_threshold_px:
                            row['merged_geometry_rejected'] += 1
                            continue
                        merged.append((i, j)); confidence.append(float(crop_scores[a]))
                        used0.add(i); used1.add(j); row['added'] += 1
        combined.append((name0, name1, merged, confidence))
        rows.append(row)

    output_features = Path(features_path).with_name(Path(features_path).stem + '_overlap.h5')
    output_matches = Path(matches_path).with_name(Path(matches_path).stem + '_overlap.h5')
    if output_features.exists() or output_matches.exists():
        raise FileExistsError('Overlap outputs already exist')
    shutil.copy2(features_path, output_features)
    with h5py.File(output_features, 'a') as features:
        for name, index in indices.items():
            for key, values in [('keypoints', np.asarray(index.points)), ('scores', np.asarray(index.scores)),
                                ('descriptors', np.asarray(index.descriptors).T)]:
                old = features[name][key]
                attributes, dtype = dict(old.attrs), old.dtype
                del features[name][key]
                dataset = features[name].create_dataset(key, data=values.astype(dtype))
                dataset.attrs.update(attributes)
    with h5py.File(output_matches, 'w') as matches_file:
        for name0, name1, merged, confidence in combined:
            mapping = np.full(len(indices[name0].points), -1, dtype=np.int32)
            scores = np.zeros(len(mapping), np.float32)
            if merged:
                a, b = np.asarray(merged).T
                mapping[a], scores[a] = b, confidence
            group = matches_file.create_group(names_to_pair(name0, name1))
            group.create_dataset('matches0', data=mapping)
            group.create_dataset('matching_scores0', data=scores)
    summary = {'pairs': len(rows), 'margin_per_side': margin, 'resize_max': resize_max,
               'geometry_threshold_px': geometry_threshold_px,
               'merged_geometry_rejected': sum(row['merged_geometry_rejected'] for row in rows),
               'seconds': time.perf_counter() - started, 'added_matches': sum(row['added'] for row in rows),
               'rows': rows}
    output_matches.with_suffix('.json').write_text(json.dumps(summary, indent=2))
    return output_features, output_matches, summary
