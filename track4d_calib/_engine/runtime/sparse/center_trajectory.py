"""Reprojection measurements and independent trajectory checks."""

import cv2
import numpy as np


def reprojection_errors(rotations, centers, intrinsics, distortions, observations):
    if len({len(x) for x in (rotations, centers, intrinsics, distortions, observations)}) != 1:
        raise ValueError("Pose and observation frame counts differ")
    result = []
    for rotation, center, k, d, row in zip(rotations, centers, intrinsics, distortions, observations):
        xyz, uv = row["xyz"], row["uv"]
        if not len(xyz):
            result.append((np.empty(0), np.empty(0, bool)))
            continue
        predicted = cv2.projectPoints(xyz - center, cv2.Rodrigues(rotation)[0], np.zeros(3), k, d)[0][:, 0]
        error = np.linalg.norm(predicted - uv, axis=1)
        valid = ((xyz - center) @ rotation.T)[:, 2] > 1e-8
        result.append((error, valid & np.isfinite(error)))
    return result


def verify_trajectory(before, after, *, max_frame_regression=.5, minimum_points=12):
    """Measure both folds; their independence is attested by the caller."""
    reports = {}
    for role in ("confirm", "final"):
        if not before[role] or len(before[role]) != len(after[role]):
            raise ValueError("Verification frame counts differ or are empty")
        rows, before_all, after_all = [], [], []
        for frame, ((old, old_valid), (new, new_valid)) in enumerate(zip(before[role], after[role])):
            if len({len(old), len(new), len(old_valid), len(new_valid)}) != 1:
                raise ValueError("Verification observation sets differ")
            enough = len(old) >= minimum_points
            row = {"frame": frame, "count": len(old), "enough": enough,
                   "new_invalid": int((old_valid & ~new_valid).sum()),
                   "before_median_px": float(np.median(old)) if len(old) else None,
                   "after_median_px": float(np.median(new)) if len(new) else None}
            row["regression_ok"] = bool(enough and row["after_median_px"] <= row["before_median_px"] + max_frame_regression)
            rows.append(row)
            before_all.extend(old.tolist())
            after_all.extend(new.tolist())
        old, new = np.array(before_all), np.array(after_all)
        valid = bool(len(old) and np.isfinite(old).all() and np.isfinite(new).all())
        summary = {"before_mean_px": float(np.mean(old)) if valid else None,
                   "after_mean_px": float(np.mean(new)) if valid else None,
                   "before_median_px": float(np.median(old)) if valid else None,
                   "after_median_px": float(np.median(new)) if valid else None,
                   "before_inliers": sum(int(((e <= 4) & v).sum()) for e, v in before[role]),
                   "after_inliers": sum(int(((e <= 4) & v).sum()) for e, v in after[role])}
        checks = {"finite": valid,
                  "mean_improved": bool(valid and np.mean(new) < np.mean(old)),
                  "median_improved": bool(valid and np.median(new) < np.median(old)),
                  "inliers_preserved": summary["after_inliers"] >= summary["before_inliers"],
                  "enough_observations": all(row["enough"] for row in rows),
                  "frame_regression_bounded": all(row["regression_ok"] for row in rows),
                  "valid_projections_preserved": not any(row["new_invalid"] for row in rows)}
        accepted = all(checks.values())
        reports[role] = {"accepted": accepted, **summary, "frames": rows, "checks": checks,
                         "failed_checks": [key for key, passed in checks.items() if not passed]}
    return {"accepted": all(row["accepted"] for row in reports.values()), "roles": reports}
