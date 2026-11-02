"""Fit continuous camera motion to image tracks around an unknown rig center."""

import cv2
import numpy as np
from scipy.interpolate import BSpline
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def spline_basis(frame_ids, spacing):
    times = np.asarray(frame_ids, dtype=float)
    if (times.ndim != 1 or len(times) < 4 or not np.isfinite(times).all()
            or np.any(np.diff(times) <= 0)):
        raise ValueError("At least four increasing finite frame times are required")
    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError("Knot spacing must be finite and positive")
    knots = np.r_[np.repeat(times[0], 4), np.arange(times[0] + spacing, times[-1], spacing),
                  np.repeat(times[-1], 4)]
    return BSpline.design_matrix(times, knots, 3).toarray()


class MotionProblem:
    """Track reprojection with independent center and rotation constraints."""

    def __init__(self, rotations, centers, intrinsics, distortions, observations,
                 frame_ids, scale, center_radius_fraction, knot_spacing):
        self.basis = np.eye(len(frame_ids))
        self.center_basis = spline_basis(frame_ids, knot_spacing)
        self.center_controls = self.center_basis.shape[1]
        self.frames, self.controls = self.basis.shape
        self.scale = float(scale)
        self.radius = float(center_radius_fraction) * self.scale
        if not np.isfinite(self.scale) or self.scale <= 0 or not np.isfinite(self.radius) or self.radius < 0:
            raise ValueError("Scale must be positive and center radius nonnegative")
        r, c, k = (np.asarray(x, dtype=float) for x in (rotations, centers, intrinsics))
        if (r.shape != (self.frames, 3, 3) or c.shape != (self.frames, 3)
                or k.shape != r.shape or len(distortions) != self.frames
                or len(observations) != self.frames):
            raise ValueError("Camera and observation counts must match frame times")
        if not all(np.isfinite(x).all() for x in (r, c, k, *distortions)):
            raise ValueError("Camera parameters must be finite")
        if (not np.allclose(r @ r.transpose(0, 2, 1), np.eye(3), atol=1e-6)
                or not np.allclose(np.linalg.det(r), 1, atol=1e-6)):
            raise ValueError("Rotations must belong to SO(3)")
        self.intrinsics, self.distortions = k, distortions
        self.observations = []
        for row in observations:
            xyz, uv = (np.asarray(row[key], dtype=float) for key in ('xyz', 'uv'))
            if (xyz.ndim != 2 or xyz.shape[1:] != (3,) or len(xyz) < 12
                    or uv.shape != (len(xyz), 2) or not np.isfinite(xyz).all() or not np.isfinite(uv).all()):
                raise ValueError("Each frame needs at least 12 finite 2D–3D observations")
            self.observations.append((xyz, uv))
        self.reference_rotation = Rotation.from_matrix(r).mean().as_matrix()
        self.reference_center = np.median(c, axis=0)
        local = Rotation.from_matrix(r @ self.reference_rotation.T).as_rotvec()
        if np.linalg.norm(local, axis=1).max() >= np.pi / 2:
            raise ValueError("Motion window exceeds the local rotation chart")
        coefficients = np.linalg.lstsq(self.basis, local, rcond=None)[0]
        self.rotation_count = 3 * self.controls
        self.moving = self.radius > 0
        offset_count = 3 * self.center_controls if self.moving else 0
        center_parameters = np.zeros(3 + offset_count)
        center_steps = np.r_[np.full(3, .01), np.ones(offset_count)]
        self.initial = np.r_[coefficients.ravel(), center_parameters]
        self.parameter_steps = np.r_[np.full(self.rotation_count, .05), center_steps]
        self.projection_rows = sum(2 * len(xyz) for xyz, _ in self.observations)

    def unpack(self, parameters):
        p = np.asarray(parameters)
        rotations = self.basis @ p[:self.rotation_count].reshape(self.controls, 3)
        center = self.reference_center + self.scale * p[self.rotation_count:self.rotation_count + 3]
        if not self.moving:
            return rotations, np.broadcast_to(center, (self.frames, 3)), center, None
        u = p[self.rotation_count + 3:].reshape(self.center_controls, 3)
        norm = np.sqrt(1 + np.sum(u * u, axis=1))
        offsets = self.radius * u / norm[:, None]
        derivative = self.radius * (np.eye(3)[None] / norm[:, None, None]
            - u[:, :, None] * u[:, None, :] / norm[:, None, None] ** 3)
        return rotations, center + self.center_basis @ offsets, center, derivative

    def linearize(self, parameters, jacobian=False):
        rotations, centers, common_center, derivative = self.unpack(parameters)
        residuals, blocks = [], []
        m = self.rotation_count
        for f, (points, pixels) in enumerate(self.observations):
            local_points = (points - centers[f]) @ self.reference_rotation.T
            translation = np.zeros(3)
            predicted, j = cv2.projectPoints(local_points, rotations[f], translation,
                                             self.intrinsics[f], self.distortions[f])
            if not jacobian:
                residuals.append((predicted[:, 0] - pixels).ravel())
                continue
            rotation = cv2.Rodrigues(rotations[f])[0] @ self.reference_rotation
            center_jacobian = -j[:, 3:6] @ rotation
            block = np.zeros((2 * len(points), len(parameters)))
            block[:, :m] = np.hstack([weight * j[:, :3] for weight in self.basis[f]])
            block[:, m:m+3] = center_jacobian * self.scale
            if self.moving:
                block[:, m+3:] = np.hstack([weight * center_jacobian @ dj
                                            for weight, dj in zip(self.center_basis[f], derivative)])
            blocks.append(block)
        if self.moving:
            # Penalizing normalized control offsets also identifies the shared center.
            if jacobian:
                count = 3 * self.center_controls
                block = np.zeros((count, len(parameters)))
                block[:, m+3:] = 2 * np.eye(count)
                blocks.append(block)
            else:
                residuals.append(2 * parameters[m+3:])
        return np.vstack(blocks) if jacobian else np.concatenate(residuals)

    def loss(self, squared):
        rho = np.vstack([squared.copy(), np.ones_like(squared), np.zeros_like(squared)])
        indices = np.flatnonzero(squared[:self.projection_rows] > 1)
        roots = np.sqrt(squared[indices])
        rho[:, indices] = np.vstack([2 * roots - 1, 1 / roots, -.5 / roots ** 3])
        return rho


def fit_camera_motion(rotations, centers, intrinsics, distortions, observations,
                      frame_ids, *, scale, center_radius_fraction=0.0001,
                      knot_spacing=10., max_evaluations=100):
    """Fit framewise rotations and a bounded smooth center trajectory."""
    problem = MotionProblem(rotations, centers, intrinsics, distortions, observations,
                            frame_ids, scale, center_radius_fraction, knot_spacing)
    solution = least_squares(problem.linearize, problem.initial,
        jac=lambda p: problem.linearize(p, True), loss=problem.loss, f_scale=2.,
        x_scale=problem.parameter_steps,
        max_nfev=max_evaluations, ftol=1e-6, xtol=1e-7, gtol=1e-6)
    local, fitted_centers, common, _ = problem.unpack(solution.x)
    fitted_rotations = Rotation.from_rotvec(local).as_matrix() @ problem.reference_rotation
    return {'rotations': fitted_rotations, 'centers': fitted_centers.copy(),
            'common_center': common, 'control_parameters': solution.x,
            'solver': {'success': bool(solution.success), 'status': int(solution.status),
                       'evaluations': int(solution.nfev), 'cost': float(solution.cost),
                       'optimality': float(solution.optimality)}}
