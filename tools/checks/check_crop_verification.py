#!/usr/bin/env python3
"""Exercise crop publication checks with disposable camera and track fixtures."""

import copy
import json
import hashlib
from pathlib import Path
import sys
import shutil
import tempfile
import unittest

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from track4d_calib.verification import verify_calibration, verify_crop_refinement
from track4d_calib.runner import _publish_poses


class CropVerificationChecks(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="track4d_calib-crop-check-")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.config = {"frames": 2, "temporal_anchor_count": 2,
                       "static_views": ["00"], "dynamic_views": ["80"]}
        self.evidence = self.root / "crop_matches"
        self.evidence.mkdir()
        self.protocol = {"anchor_crop": True, "anchors": [0, 1], "frames": [0, 1]}
        self.result = {"committed": 1, "rows": []}
        self.observations = {}
        for role in range(3):
            ids = np.arange(role * 20, (role + 1) * 20)
            xyz = np.column_stack((ids * 0.01, (ids % 5) * 0.02, np.full(20, 5.)))
            uv = xyz[:, :2] / 5 * 1000 + 256
            for key, values in (("ids", ids), ("xyz", xyz), ("uv", uv)):
                self.observations[f"{key}_{role}"] = values
        for frame in range(2):
            self.write_cameras("pre_crop_poses", frame, 0.01)
            self.write_cameras("poses", frame, 0. if frame == 0 else 0.01)
            np.savez_compressed(self.observation_path(frame), **self.observations)
            self.result["rows"].append({
                "frame": frame, "view": "80", "committed": frame == 0,
                "role_counts": [20, 20, 20],
                "before": self.scores(2.), "after": self.scores(0.),
                "single_pass": self.scores(1.),
            })
        self.save_records()

    @staticmethod
    def scores(error):
        return [{"count": 20, "inliers": 20, "median_px": error} for _ in range(2)]

    def observation_path(self, frame=0):
        return self.evidence / f"f{frame:06d}_v80.npz"

    def write_cameras(self, stage, frame, translation, static_translation=0., distortion=0.):
        folder = self.root / stage / f"f{frame:06d}/cameras_pred"
        folder.mkdir(parents=True, exist_ok=True)
        for filename in ("intri.yml", "extri.yml"):
            storage = cv2.FileStorage(str(folder / filename), cv2.FILE_STORAGE_WRITE)
            try:
                for view, offset in (("00", static_translation), ("80", translation)):
                    if filename == "intri.yml":
                        storage.write("K_" + view, np.array([[1000., 0, 256], [0, 1000., 256], [0, 0, 1]]))
                        storage.write("D_" + view, np.array([distortion, 0, 0, 0, 0.]))
                    else:
                        storage.write("Rot_" + view, np.eye(3))
                        storage.write("T_" + view, np.array([[offset], [0.], [0.]]))
            finally:
                storage.release()

    def save_records(self):
        (self.evidence / "PROTOCOL.json").write_text(json.dumps(self.protocol))
        (self.evidence / "RESULT.json").write_text(json.dumps(self.result))

    def check_rejected(self):
        with self.assertRaises((ValueError, FileNotFoundError)):
            verify_crop_refinement(self.root, self.config)

    def test_valid_and_read_only(self):
        before = {p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        report = verify_crop_refinement(self.root, self.config)
        self.assertEqual((report["crop_nodes"], report["committed_nodes"], report["camera_nodes"]), (2, 1, 4))
        self.assertEqual(before, {p: p.read_bytes() for p in before})

    def test_disabled_stages(self):
        for options in ({"dynamic_overlap_crop": False},):
            self.assertEqual(verify_crop_refinement(self.root / "absent", dict(self.config, **options)),
                             {"status": "disabled"})

    def test_missing_evidence(self):
        self.observation_path().unlink()
        self.check_rejected()

    def test_missing_anchor(self):
        self.protocol["frames"] = [0]
        self.save_records()
        self.check_rejected()

    def test_duplicate_rows(self):
        self.result["rows"][1] = copy.deepcopy(self.result["rows"][0])
        self.save_records()
        self.check_rejected()

    def test_wrong_commit_count(self):
        self.result["committed"] = 0
        self.save_records()
        self.check_rejected()

    def test_missing_single_pass(self):
        self.result["rows"][0]["single_pass"] = []
        self.save_records()
        self.check_rejected()

    def test_overlapping_roles(self):
        self.observations["ids_1"][0] = self.observations["ids_0"][0]
        np.savez_compressed(self.observation_path(), **self.observations)
        self.check_rejected()

    def test_duplicate_point_ids(self):
        self.observations["ids_1"][1] = self.observations["ids_1"][0]
        np.savez_compressed(self.observation_path(), **self.observations)
        self.check_rejected()

    def test_nonfinite_observations(self):
        self.observations["uv_1"][0, 0] = np.nan
        np.savez_compressed(self.observation_path(), **self.observations)
        self.check_rejected()

    def test_malformed_observations(self):
        self.observations["xyz_1"] = self.observations["xyz_1"][:, :2]
        np.savez_compressed(self.observation_path(), **self.observations)
        self.check_rejected()

    def test_false_recorded_scores(self):
        for value in (1., float("nan"), float("inf")):
            with self.subTest(value=value):
                self.result["rows"][0]["before"][0]["median_px"] = value
                self.save_records()
                self.check_rejected()

    def test_truncated_scores(self):
        self.result["rows"][0]["after"] = self.scores(0.)[:1]
        self.save_records()
        self.check_rejected()

    def test_no_geometric_improvement(self):
        self.write_cameras("poses", 0, 0.01)
        self.result["rows"][0]["after"] = self.scores(2.)
        self.result["rows"][0]["single_pass"] = self.scores(3.)
        self.save_records()
        self.check_rejected()

    def test_changed_final_pose(self):
        self.write_cameras("poses", 0, 0.005)
        self.check_rejected()

    def test_changed_uncommitted_pose(self):
        self.write_cameras("poses", 1, 0.02)
        self.check_rejected()

    def test_changed_static_pose(self):
        self.write_cameras("poses", 0, 0., static_translation=0.01)
        self.check_rejected()

    def test_changed_intrinsics(self):
        self.write_cameras("poses", 0, 0., distortion=0.01)
        self.check_rejected()

    def test_publication_and_readout(self):
        with tempfile.TemporaryDirectory(prefix="track4d_calib-publication-check-") as directory:
            output = Path(directory)
            calibration = output / "calibration"
            shutil.copytree(self.root, calibration)
            config = dict(self.config, center_radius=None)
            (calibration / "config.json").write_text(json.dumps(config))
            hashes = {str(p.relative_to(calibration / "poses")):
                      hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in (calibration / "poses").glob("f*/cameras_pred/*.yml")}
            (calibration / "RESULT.json").write_text(json.dumps({
                "registered": 2, "pose_hashes": hashes, "wall_seconds": 1., "stages": []}))
            (calibration / "CLOSURE.json").write_text(json.dumps({
                "rows": [{"committed": False, "base_confirm": self.scores(2.)[0]} for _ in range(2)]}))
            seal, published = _publish_poses(output, calibration, 2)
            self.assertEqual(hashes, published)
            (output / "result.json").write_text(json.dumps(dict(
                seal, pose_sha256=hashes, input_audit_passed=True, pose_grid_complete=True)))
            (output / "protocol.json").write_text(json.dumps({"config": config, "code_sha256": {}}))
            (output / "input_audit.json").write_text('{"passed": true}')
            runtime = calibration / "runtime_scene"
            static = runtime / "cameras/static"
            static.mkdir(parents=True)
            rig_hashes = {}
            for name in ("intri", "extri"):
                shutil.copy2(calibration / f"poses/f000000/cameras_pred/{name}.yml", static / f"{name}.yml")
                rig_hashes[name + "_sha256"] = hashlib.sha256((static / f"{name}.yml").read_bytes()).hexdigest()
            (runtime / "runtime_manifest.json").write_text(json.dumps({"fresh_static_rig": rig_hashes}))
            verify_calibration(output)

            evidence = calibration / "crop_matches/RESULT.json"
            record = json.loads(evidence.read_text())
            record["seconds"] = 99.
            evidence.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, "evidence changed"):
                verify_calibration(output)
            record["rows"][0]["before"][0]["median_px"] = 1.
            evidence.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, "scores differ"):
                verify_calibration(output)
            (output / "poses").unlink()
            (calibration / "POSE_AUDIT.json").unlink()
            with self.assertRaises(ValueError):
                _publish_poses(output, calibration, 2)
            self.assertFalse((output / "poses").exists())
            self.assertFalse((calibration / "POSE_AUDIT.json").exists())


if __name__ == "__main__":
    unittest.main()
