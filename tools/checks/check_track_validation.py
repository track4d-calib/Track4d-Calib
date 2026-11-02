#!/usr/bin/env python3
"""Check persistent observation isolation and camera-motion parameterizations."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "track4d_calib/_engine/runtime/sparse"))
from track_partition import identity_roles, run, frame_roles, fitting_track_root, exclude_reserved_queries
from camera_motion import MotionProblem, fit_camera_motion
from center_trajectory import verify_trajectory


class TrackPartitionTests(unittest.TestCase):
    def test_stable_across_time_order_and_subsets(self):
        ids = np.arange(1000)
        roles = identity_roles(ids)
        np.testing.assert_array_equal(identity_roles(ids[::-1]), roles[::-1])
        np.testing.assert_array_equal(identity_roles(ids[::3]), roles[::3])
        self.assertEqual(set(roles), {0, 1, 2})

    def test_filtered_visibility_and_crop_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text(json.dumps({"frames": 2, "track_validation": {"exclusion_radius_px": 2.}}))
            (root / "anchor_tracks").mkdir()
            (root / "anchor_tracks/PROTOCOL.json").write_text('{"anchors":[0]}')
            points = np.stack([np.arange(30)*10., np.zeros(30)], axis=-1)[None, None]
            for name in ("anchor_tracks/f000000", "cotracker_cached_tracks/f000000", "cotracker_cached_tracks/f000001"):
                folder = root / name; folder.mkdir(parents=True)
                np.save(folder / "tracks.npy", points)
                np.save(folder / "visibility.npy", np.ones((1, 1, 30), bool))
                (folder / "views.txt").write_text("81\n")
                (folder / "meta.json").write_text(json.dumps({"point_ids": list(range(30))}))
            with self.assertRaises(ValueError):
                fitting_track_root(root)
            run(root)
            roles = frame_roles(root, 0)
            np.testing.assert_array_equal(roles, frame_roles(root, 1))
            original = np.load(root / "cotracker_cached_tracks/f000000/visibility.npy")
            filtered = np.load(fitting_track_root(root) / "f000000/visibility.npy")
            self.assertTrue(original.all())
            self.assertFalse(filtered[..., roles != 0].any())
            self.assertTrue(filtered[..., roles == 0].all())
            allowed = exclude_reserved_queries(root, 0, "81", points[0, 0])
            np.testing.assert_array_equal(allowed, roles == 0)
            self.assertFalse(exclude_reserved_queries(root, 0, "81", points[0, 0][roles != 0] + [1., 0.]).any())


def fixture():
    rng = np.random.default_rng(8)
    count = 5
    r = np.stack([cv2.Rodrigues(np.array([.01*f, -.02*f, .005*f]))[0] for f in range(count)])
    c = rng.normal(0, .01, (count, 3))
    k = np.tile(np.array([[500., 0, 320.], [0, 500., 240.], [0, 0, 1.]]), (count, 1, 1))
    d = np.zeros((count, 5)); observations = []
    for i in range(count):
        xyz = rng.uniform([-1, -1, 3], [1, 1, 6], (24, 3))
        uv = cv2.projectPoints(xyz-c[i], cv2.Rodrigues(r[i])[0], np.zeros(3), k[i], d[i])[0][:, 0]
        observations.append({"xyz": xyz, "uv": uv})
    return r, c, k, d, observations, np.arange(count)


class MotionTests(unittest.TestCase):
    def test_jacobians_and_parameter_constraints(self):
        args = fixture()
        for radius in (0., .0001, .01):
            with self.subTest(radius=radius):
                problem = MotionProblem(*args, scale=5., center_radius_fraction=radius, knot_spacing=3.)
                parameters = problem.initial.copy()
                parameters[problem.rotation_count + 3:] = .3
                step = 1e-6
                numeric = np.column_stack([
                    (problem.linearize(parameters + np.eye(len(parameters))[j]*step) -
                     problem.linearize(parameters - np.eye(len(parameters))[j]*step))/(2*step)
                    for j in range(len(parameters))])
                np.testing.assert_allclose(problem.linearize(parameters, True), numeric, rtol=2e-5, atol=1e-5)
                _, centers, common, _ = problem.unpack(parameters)
                self.assertLessEqual(np.max(np.linalg.norm(centers-common, axis=1)), radius*5.)
                altered = parameters.copy()
                altered[0] += .1
                np.testing.assert_array_equal(centers, problem.unpack(altered)[1])

    def test_verification_failure_is_explicit(self):
        old = {r: [(np.full(12, 2.), np.ones(12, bool))] for r in ("confirm", "final")}
        good = {r: [(np.full(12, 1.), np.ones(12, bool))] for r in old}
        self.assertTrue(verify_trajectory(old, good)["accepted"])
        same = verify_trajectory(old, old)
        self.assertFalse(same["accepted"])
        self.assertEqual(same["roles"]["final"]["failed_checks"], ["mean_improved", "median_improved"])


if __name__ == "__main__":
    unittest.main()
