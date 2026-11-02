"""Add overlap-crop observations without changing existing query correspondences."""
import time
import numpy as np
from PIL import Image


def overlap_bounds(points, image_size, margin=0.1):
    height, width = image_size
    low, high = np.min(points, axis=0), np.max(points, axis=0)
    padding = (high - low) * margin
    low = np.maximum(0, np.floor(low - padding)).astype(int)
    high = np.minimum([width, height], np.ceil(high + padding) + 1).astype(int)
    return (*map(int, low), *map(int, high))


def augment_queries(*, matcher, model, first, source_image, target_image,
                    queries, height, width, minimum_overlap=0.2,
                    maximum_cycle_error_px=1.5, crop_source=True):
    """Keep original query IDs and append bidirectionally verified observations."""
    report = {"first_count": len(first["point_ids"]), "added": 0, "crop_seconds": 0.0}
    result = dict(first)
    result["crop_diagnostics"] = report
    if len(first["point_ids"]) < 16:
        report["status"] = "insufficient_overlap"
        return result
    source_bounds = (overlap_bounds(first["static_points"], (height, width))
                     if crop_source else (0, 0, width, height))
    target_bounds = overlap_bounds(first["target_points"], (height, width))
    report["bounds"] = [source_bounds, target_bounds]
    if any(min(b[2] - b[0], b[3] - b[1]) < 32 for b in [source_bounds, target_bounds]):
        report["status"] = "small_crop"
        return result
    started = time.perf_counter()
    crops = []
    for path, bounds in [(source_image, source_bounds), (target_image, target_bounds)]:
        with Image.open(path) as image:
            if image.size != (width, height):
                raise ValueError("Image size differs from query coordinates")
            crops.append(image.convert("RGB").crop(bounds))
    inside = ((queries[:, 0] >= source_bounds[0]) & (queries[:, 0] < source_bounds[2])
              & (queries[:, 1] >= source_bounds[1]) & (queries[:, 1] < source_bounds[3]))
    ids = np.flatnonzero(inside)
    prediction = model.match(*crops)
    arrays = matcher.arrays_from_prediction(prediction)
    relay = matcher.relay_bidirectional_query_points(
        source_points_xy=queries[ids] - np.array(source_bounds[:2]),
        warp_source_to_target=arrays["warp_AB"],
        warp_target_to_source=arrays["warp_BA"],
        overlap_source_to_target=arrays["overlap_AB"],
        overlap_target_to_source=arrays["overlap_BA"],
        source_image_size=(crops[0].height, crops[0].width),
        target_image_size=(crops[1].height, crops[1].width),
        minimum_overlap=minimum_overlap,
        maximum_cycle_error_px=maximum_cycle_error_px,
    )
    valid = np.asarray(relay["valid"], dtype=bool)
    report["second_valid"] = int(valid.sum())
    valid &= ~np.isin(ids, first["point_ids"])
    extra = {
        "point_ids": ids[valid],
        "static_points": queries[ids[valid]],
        "target_points": np.asarray(relay["target_points"])[valid] + np.array(target_bounds[:2]),
        **{key: np.asarray(relay[key])[valid] for key in
           ["forward_overlap", "backward_overlap", "cycle_error_px"]},
    }
    order = np.argsort(np.concatenate([first["point_ids"], extra["point_ids"]]))
    for key in extra:
        result[key] = np.concatenate([first[key], extra[key]], axis=0)[order]
    report.update(status="complete", added=int(valid.sum()), crop_seconds=time.perf_counter()-started)
    result["inference_seconds"] = first["inference_seconds"] + report["crop_seconds"]
    return result


def geometry_admitted_points(ids, xyz, first, augmented, cameras, *, triangulate):
    """Add observations consistent with independent original support geometry."""
    import cv2

    filtered = {view: dict(points) for view, points in first.items()}
    offered = accepted = 0
    for view, camera in cameras.items():
        if view not in augmented:
            continue
        projected, _ = cv2.projectPoints(
            xyz, cv2.Rodrigues(camera.R)[0], camera.T, camera.K, camera.D
        )
        depth = (xyz @ camera.R.T + camera.T.reshape(1, 3))[:, 2]
        for index, point_id in enumerate(ids):
            if point_id in augmented[view] and point_id not in first[view]:
                offered += 1
                pixel = augmented[view][point_id]
                if depth[index] > 0 and np.linalg.norm(projected[index, 0] - pixel) <= 2.5:
                    filtered[view][point_id] = pixel
                    accepted += 1
    points = []
    fallback = 0
    for point_id, original in zip(ids, xyz):
        observations = [
            (view, cameras[view], mapping[point_id])
            for view, mapping in filtered.items() if point_id in mapping
        ]
        result = triangulate(observations)
        points.append(result["point"] if result["success"] else original)
        fallback += not result["success"]
    return np.asarray(points, dtype=float).reshape(-1, 3), {
        "offered": offered, "accepted": accepted, "triangulation_fallback": fallback,
    }
