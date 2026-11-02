#!/usr/bin/env python3
"""Compare batched RGB triangulation with the scalar geometric oracle."""

from pathlib import Path
import sys
import unittest

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ENGINE = ROOT / 'track4d_calib/_engine/runtime'
sys.path[:0] = [str(ENGINE / 'sparse'), str(ENGINE / 'components/track_closure/geometry')]
from track4d_calib.cameras import CameraParams
from track4d_calib.same_time_multiview_pnp import SameTimePnPConfig, _triangulate_robust
from static_triangulation import triangulate_queries


class StaticTriangulationTests(unittest.TestCase):
    def compare(self, cameras, mappings, allowed, config):
        expected_ids, expected_points = [], []
        for query in np.flatnonzero(allowed):
            result = _triangulate_robust(observations=[(str(i), camera, mapping[query])
                for i, (camera, mapping) in enumerate(zip(cameras, mappings)) if query in mapping], config=config)
            if result['success']:
                expected_ids.append(query)
                expected_points.append(result['point'])
        ids, points = triangulate_queries(cameras, mappings, allowed, config)
        np.testing.assert_array_equal(ids, expected_ids)
        np.testing.assert_allclose(points, np.asarray(expected_points).reshape(-1, 3), rtol=0., atol=1e-10)
        return ids

    def fixture(self, distorted):
        rng = np.random.default_rng(31)
        points = rng.uniform([-1, -.8, 3], [1, .8, 6], (96, 3))
        points[80:, 2] *= -1
        cameras, mappings = [], []
        for i in range(8):
            center = np.array([np.cos(i * np.pi / 4), .6 * np.sin(i * np.pi / 4), 0.])
            rotation = cv2.Rodrigues(np.array([.03 * i, -.02 * i, 0.]))[0]
            camera = CameraParams(str(i), np.array([[520., 0, 320.], [0, 510., 240.], [0, 0, 1.]]),
                rotation, -rotation @ center, np.array([.08, -.015, .001, -.002, 0.]) if distorted else np.zeros(5), 480, 640)
            uv = cv2.projectPoints(points, cv2.Rodrigues(rotation)[0], camera.T, camera.K, camera.D)[0].reshape(-1, 2)
            if distorted:
                uv += rng.normal(0, .3, uv.shape)
                uv[i::9] += [35., -21.]
            cameras.append(camera)
            mappings.append({q: pixel for q, pixel in enumerate(uv) if (q + i) % 7 != 0})
        allowed = np.arange(len(points)) % 5 != 0
        return cameras, mappings, allowed

    def test_noisy_distorted_tracks_and_missing_supports(self):
        ids = self.compare(*self.fixture(True), SameTimePnPConfig(support_reprojection_error_px=2.5,
                            min_triangulation_angle_deg=.5))
        self.assertGreater(len(ids), 40)
        self.assertTrue(np.all(ids < 80))

    def test_exact_tracks_and_minimum_support_count(self):
        for minimum in (2, 4, 7):
            with self.subTest(minimum=minimum):
                self.compare(*self.fixture(False), SameTimePnPConfig(min_support_views=minimum))

    def test_degenerate_baselines_and_empty_queries(self):
        cameras, mappings, allowed = self.fixture(False)
        self.compare(cameras, [{} for _ in cameras], allowed, SameTimePnPConfig())
        self.compare(cameras, mappings, np.zeros(len(allowed), dtype=bool), SameTimePnPConfig())
        self.compare([cameras[0]] * len(cameras), mappings, allowed, SameTimePnPConfig())


if __name__ == '__main__':
    unittest.main()
