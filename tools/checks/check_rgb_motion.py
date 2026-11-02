#!/usr/bin/env python3
"""Check RGB observation isolation, spatial coverage and strict frame guards."""

from pathlib import Path
import sys
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'track4d_calib/_engine/runtime/sparse'))
from rgb_observations import spatial_sample, separated
from refine_rgb_motion import audit_roles
from center_trajectory import verify_trajectory
from track_partition import identity_roles
from motion_verification import center_high_frequency, verify_camera_motion


class RGBMotionTests(unittest.TestCase):
    def test_sampling_retains_spatial_support_under_reordering(self):
        rng = np.random.default_rng(4)
        uv = np.r_[rng.normal(0, .01, (200, 2)), [[-2, -2], [2, -2], [2, 2], [-2, 2]]]
        ids = np.arange(len(uv))
        keep = spatial_sample(uv, ids, 12)
        self.assertTrue(set(range(200, 204)) <= set(keep))
        permutation = rng.permutation(len(uv))
        again = spatial_sample(uv[permutation], ids[permutation], 12)
        np.testing.assert_array_equal(ids[keep], ids[permutation][again])

    def test_role_guard_rejects_cross_time_reuse_and_nearby_queries(self):
        ids = np.arange(100)
        roles = identity_roles(ids)
        row = {name + '_ids': ids[roles == number] for number, name in enumerate(('fit', 'confirm', 'final'))}
        audit_roles([row, row])
        changed = dict(row, fit_ids=np.r_[row['fit_ids'], row['final_ids'][0]])
        with self.assertRaises(ValueError):
            audit_roles([row, changed])
        np.testing.assert_array_equal(separated(np.array([[0., 0.], [2., 0.], [2.01, 0.]]), np.array([[0., 0.]]), 2.), [False, False, True])

    def test_global_gain_does_not_hide_a_regressed_frame(self):
        before = {role: [(np.full(20, 3.), np.ones(20, bool)) for _ in range(30)] for role in ('confirm', 'final')}
        after = {role: [(np.full(20, 1.), np.ones(20, bool)) for _ in range(30)] for role in before}
        self.assertTrue(verify_trajectory(before, after)['accepted'])
        after['confirm'][7] = np.full(20, 3.6), np.ones(20, bool)
        result = verify_trajectory(before, after)
        self.assertFalse(result['accepted'])
        self.assertTrue(result['roles']['confirm']['checks']['mean_improved'])
        self.assertEqual(result['roles']['confirm']['failed_checks'], ['frame_regression_bounded'])


class MotionGuardTests(unittest.TestCase):
    def test_translation_and_smooth_acceleration_are_not_bounded(self):
        t = np.arange(30, dtype=float)
        centers = np.c_[10*t, .2*t*t, np.zeros_like(t)]
        self.assertLess(center_high_frequency(centers, t, 5.), 1e-10)
        jittered = centers + np.c_[.01*(-1.)**t, np.zeros((len(t), 2))]
        self.assertGreater(center_high_frequency(jittered, t, 5.), .001)
        rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        moved = 3*jittered@rotation.T + [10., 20., -8.]
        self.assertAlmostEqual(center_high_frequency(jittered, t, 5.),
                               center_high_frequency(moved, t, 15.), places=12)

    def test_new_jitter_blocks_an_otherwise_valid_geometric_gain(self):
        t = np.arange(30)
        before = {role: [(np.full(20, 3.), np.ones(20, bool)) for _ in t] for role in ('confirm', 'final')}
        after = {role: [(np.full(20, 1.), np.ones(20, bool)) for _ in t] for role in before}
        centers = np.c_[.1*t, np.zeros((len(t), 2))]
        jittered = centers + np.c_[.001*(-1.)**t, np.zeros((len(t), 2))]
        bad = verify_camera_motion(before, after, centers, jittered, t, 5.)
        self.assertTrue(bad['geometry_accepted'])
        self.assertFalse(bad['accepted'])
        good = verify_camera_motion(before, after, jittered, centers, t, 5.)
        self.assertTrue(good['accepted'])


if __name__ == '__main__':
    unittest.main()
