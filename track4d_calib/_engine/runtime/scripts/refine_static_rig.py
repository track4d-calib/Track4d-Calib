"""Refine a static camera rig and verify it on separate timestamps."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation

from build_static_rig import write_runtime_checksums
from feature_cache import cache_image_features

HERE = Path(__file__).resolve().parent
COMPONENT = HERE.parent / 'components/track_closure'
for path in [COMPONENT, COMPONENT / 'geometry']:
    sys.path.insert(0, str(path))
from track4d_calib.cameras import CameraLoader, CameraParams, save_cameras
from track4d_calib.global_track_ba import GlobalTrackBAConfig, _gather_triangulated_observations


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect_tracks(args, views, frames, root):
    """Build multiview query banks independently at each selected timestamp."""
    from rig_contract import make_rig_contract
    from gpu_timing import recorder
    from run_relay import install_roma_cache
    manifest = json.loads((args.runtime / 'runtime_manifest.json').read_text())
    rig = root / 'RIG_CONTRACT.json'
    write_json(rig, make_rig_contract(views, manifest['long_views'], center_motion_radius=None))
    os.environ['TRACK4D_CALIB_RIG_CONTRACT'] = str(rig)
    os.environ['TRACK4D_CALIB_NUM_FRAMES'] = str(args.num_frames)
    from scripts.geometry import export_static_pnp_inlier_seeded_track_closure as factory
    service = recorder()
    if service is not None:
        install_roma_cache(factory, service, retain=False)
    path = COMPONENT / 'chain/build_second_roma_component_anchor_cache.py'
    spec = importlib.util.spec_from_file_location('static_rig_matcher', path)
    matcher = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = matcher
    spec.loader.exec_module(matcher)
    import torch
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    model, _ = matcher._load_roma_model(source_root=args.romav2_root, weights=args.romav2_weights,
        device='cuda', setting=args.matching_quality, force_bidirectional=True)
    cache_image_features(model, capacity=64)
    cameras = CameraLoader(args.runtime / 'cameras/static').load_cameras(views)
    camera = cameras[views[0]]
    queries = matcher.query_grid(height=camera.H, width=camera.W, stride=8)
    references = [views[i * len(views) // 4] for i in range(4)]
    banks = {}
    for frame in frames:
        points_by_reference, visibility_by_reference = [], []
        for reference in references:
            points = np.zeros((len(views), len(queries), 2), dtype=np.float32)
            visible = np.zeros((len(views), len(queries)), dtype=bool)
            points[views.index(reference)] = queries
            visible[views.index(reference)] = True
            for index, view in enumerate(views):
                if view == reference:
                    continue
                match = matcher.match_queries(model=model,
                    static_image=args.runtime / f'images/{reference}/{frame:06d}.png',
                    target_image=args.runtime / f'images/{view}/{frame:06d}.png', queries=queries,
                    height=camera.H, width=camera.W, minimum_overlap=0.2, maximum_cycle_error_px=1.5)
                ids = np.asarray(match['point_ids'], dtype=np.int64)
                points[index, ids] = match['target_points']
                visible[index, ids] = True
            keep = visible.sum(axis=0) >= 3
            points_by_reference.append(points[:, keep])
            visibility_by_reference.append(visible[:, keep])
        banks[frame] = (np.concatenate(points_by_reference, axis=1), np.concatenate(visibility_by_reference, axis=1))
    np.savez_compressed(root / 'tracks.npz', views=np.asarray(views),
        **{f'tracks_{f}': bank[0] for f, bank in banks.items()},
        **{f'visibility_{f}': bank[1] for f, bank in banks.items()})
    del model
    torch.cuda.empty_cache()
    return banks


def solve_bundle(cameras, views, tracks, visibility):
    """Jointly fit static poses and points while fixing the similarity gauge."""
    config = GlobalTrackBAConfig(min_track_support_frames=1, max_tracks_total=5000)
    observations = _gather_triangulated_observations(tracks=tracks[:, None], visibility=visibility[:, None],
        view_ids=views, cameras=cameras, cfg=config)
    xyz = np.asarray(observations['points_init'])
    view_ids = np.asarray(observations['obs_view_idx'], dtype=int)
    point_ids = np.asarray(observations['obs_point_idx'], dtype=int)
    pixels = np.asarray(observations['obs_uv'])
    if len(xyz) < 16 or len(pixels) < 64 or set(view_ids) != set(range(len(views))):
        return None, {'success': False, 'reason': 'insufficient_static_track_support'}
    reference = cameras[views[0]]
    rotation0, center0 = reference.R, reference.camera_center
    rotations = np.stack([cameras[v].R @ rotation0.T for v in views])
    translations = np.stack([cameras[v].T + cameras[v].R @ center0 for v in views])
    parameters = np.concatenate([Rotation.from_matrix(rotations).as_rotvec(), translations], axis=1)
    xyz = (xyz - center0) @ rotation0.T
    farthest = int(np.argmax(np.linalg.norm(translations, axis=1)))
    coordinate = 3 + int(np.argmax(np.abs(translations[farthest])))
    free = np.ones(parameters.shape, dtype=bool)
    free[0] = False
    free[farthest, coordinate] = False
    columns = np.full(free.shape, -1, dtype=int)
    columns[free] = np.arange(free.sum())
    camera_count = int(free.sum())
    initial = np.r_[parameters[free], xyz.reshape(-1)]
    intrinsics = np.stack([cameras[v].K for v in views])
    pattern = lil_matrix((len(pixels) * 2, len(initial)), dtype=np.int8)
    for index, (view, point) in enumerate(zip(view_ids, point_ids, strict=True)):
        selected = columns[view][columns[view] >= 0]
        point_columns = camera_count + 3 * point + np.arange(3)
        pattern[2 * index:2 * index + 2, np.r_[selected, point_columns]] = 1

    def unpack(values):
        poses = parameters.copy()
        poses[free] = values[:camera_count]
        return poses, values[camera_count:].reshape(-1, 3)

    def residual(values):
        poses, points = unpack(values)
        matrices = Rotation.from_rotvec(poses[:, :3]).as_matrix()
        local = np.einsum('nij,nj->ni', matrices[view_ids], points[point_ids]) + poses[view_ids, 3:]
        projected = np.einsum('nij,nj->ni', intrinsics[view_ids], local)
        return (projected[:, :2] / np.maximum(projected[:, 2:3], 1e-8) - pixels).reshape(-1)

    before = np.linalg.norm(residual(initial).reshape(-1, 2), axis=1)
    solved = least_squares(residual, initial, jac_sparsity=pattern.tocsr(), method='trf',
        jac='3-point', x_scale='jac', loss='soft_l1', f_scale=3.0,
        max_nfev=100, ftol=1e-6, xtol=1e-6, gtol=1e-6)
    poses, _ = unpack(solved.x)
    after = np.linalg.norm(residual(solved.x).reshape(-1, 2), axis=1)
    candidate = {}
    for index, view in enumerate(views):
        rotation = Rotation.from_rotvec(poses[index, :3]).as_matrix() @ rotation0
        translation = poses[index, 3:] - rotation @ center0
        camera = cameras[view]
        candidate[view] = CameraParams(view, camera.K, rotation, translation, camera.D, camera.H, camera.W)
    report = {'success': bool(solved.success), 'message': solved.message, 'nfev': solved.nfev,
        'optimality': float(solved.optimality), 'points': len(xyz), 'observations': len(pixels),
        'mean_before_px': float(before.mean()), 'mean_after_px': float(after.mean()),
        'median_before_px': float(np.median(before)), 'median_after_px': float(np.median(after))}
    return candidate, report


def heldout_errors(cameras, tracks, visibility, views):
    """Reproject each observation from points triangulated without that camera."""
    projections = np.stack([cameras[v].P for v in views])
    errors = []
    for point in range(tracks.shape[1]):
        seen = np.flatnonzero(visibility[:, point])
        if len(seen) < 3:
            continue
        for target in seen:
            support = seen[seen != target]
            pixels, matrices = tracks[support, point], projections[support]
            rows = np.stack([pixels[:, 0, None] * matrices[:, 2] - matrices[:, 0],
                             pixels[:, 1, None] * matrices[:, 2] - matrices[:, 1]], axis=1).reshape(-1, 4)
            _, _, vt = np.linalg.svd(rows, full_matrices=False)
            homogeneous = vt[-1]
            if abs(homogeneous[3]) < 1e-12:
                errors.append(float('inf'))
                continue
            homogeneous /= homogeneous[3]
            projected = projections[seen] @ homogeneous
            if np.any(projected[:, 2] <= 1e-8):
                errors.append(float('inf'))
                continue
            projected = projections[target] @ homogeneous
            errors.append(float(np.linalg.norm(projected[:2] / projected[2] - tracks[target, point])))
    return np.asarray(errors)


def verify_bundle(original, candidate, banks, frames, views):
    """Require paired error improvement without losing valid held-out observations."""
    reports = {}
    for frame in frames:
        tracks, visibility = banks[frame]
        before = heldout_errors(original, tracks, visibility, views)
        after = heldout_errors(candidate, tracks, visibility, views)
        if before.shape != after.shape:
            raise ValueError('Verification observation identities changed')
        valid = np.isfinite(before)
        complete = bool(valid.any() and np.isfinite(after[valid]).all())
        mean_gain = float(np.mean(before[valid] - after[valid])) if complete else None
        median_gain = float(np.median(before[valid]) - np.median(after[valid])) if complete else None
        reports[str(frame)] = {'count': int(valid.sum()),
            'newly_invalid': int(np.sum(valid & ~np.isfinite(after))),
            'mean_gain_px': mean_gain, 'median_gain_px': median_gain,
            'passed': bool(complete and mean_gain > 1e-6 and median_gain > 1e-6)}
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--num-frames', type=int, choices=[30, 120], required=True)
    parser.add_argument('--romav2-root', type=Path, required=True)
    parser.add_argument('--romav2-weights', type=Path, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--matching-quality', choices=['fast', 'precise'], default='fast')
    parser.add_argument('--fit-frames', type=int, nargs='+', default=[0])
    args = parser.parse_args()
    verification_frames = [(args.num_frames - 1) // 2, args.num_frames - 1]
    if (args.fit_frames != sorted(set(args.fit_frames))
            or any(f < 0 or f >= args.num_frames for f in args.fit_frames)
            or set(args.fit_frames).intersection(verification_frames)):
        parser.error('fit frames must be ordered, unique and separate from verification frames')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    args.root.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((args.runtime / 'runtime_manifest.json').read_text())
    views = list(map(str, manifest['public_static_views']))
    installed = args.runtime / 'cameras/static'
    original = CameraLoader(installed).load_cameras(views)
    if any(np.count_nonzero(camera.D) for camera in original.values()):
        raise ValueError('Static rig refinement requires undistorted pinhole images')
    frames = sorted([*args.fit_frames, *verification_frames])
    pairs = len(frames) * 4 * (len(views) - 1)
    write_json(args.root / 'PROTOCOL.json', {'static_views': views, 'fit_frames': args.fit_frames,
        'verification_frames': verification_frames, 'matching_pairs': pairs, 'query_stride': 8,
        'matching_quality': args.matching_quality,
        'original_pose_sha256': {p.name: digest(p) for p in installed.glob('*.yml')},
        'frozen_before_matching': True})
    banks = collect_tracks(args, views, frames, args.root)
    tracks = np.concatenate([banks[f][0] for f in args.fit_frames], axis=1)
    visibility = np.concatenate([banks[f][1] for f in args.fit_frames], axis=1)
    candidate, solver = solve_bundle(original, views, tracks, visibility)
    checks = verify_bundle(original, candidate, banks, verification_frames, views) if candidate is not None else {}
    accepted = bool(candidate is not None and solver['success'] and checks
                    and all(row['passed'] for row in checks.values())
                    and solver['mean_before_px'] - solver['mean_after_px'] > 1e-6
                    and solver['median_before_px'] - solver['median_after_px'] > 1e-6)
    if candidate is not None:
        save_cameras(args.root / 'candidate_cameras', candidate)
    if accepted:
        for name in ['intri.yml', 'extri.yml']:
            temporary = installed / ('.refined_' + name)
            shutil.copy2(args.root / 'candidate_cameras' / name, temporary)
            temporary.replace(installed / name)
    write_json(args.root / 'RESULT.json', {'status': 'complete', 'accepted': accepted,
        'matching_pairs': pairs, 'solver': solver, 'verification': checks,
        'installed_pose_sha256': {p.name: digest(p) for p in installed.glob('*.yml')}})
    manifest['fresh_static_rig'].update({
        'intri_sha256': digest(installed / 'intri.yml'),
        'extri_sha256': digest(installed / 'extri.yml'),
        'refinement': {'path': str(args.root / 'RESULT.json'),
                       'sha256': digest(args.root / 'RESULT.json')},
    })
    write_json(args.runtime / 'runtime_manifest.json', manifest)
    write_runtime_checksums(args.runtime, manifest['protocol'])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
