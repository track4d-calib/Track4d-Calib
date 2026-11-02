#!/usr/bin/env python3
"""Check motion configuration and the calibration stage lifecycle on CPU."""

import json
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from track4d_calib.config import load_config, validate_config
from track4d_calib.evaluation import snapshot_verification


class MotionConfigTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(ROOT / "configs/huahua.json", level="quality")

    def test_partition_is_resolved_before_pnp_and_config_is_idempotent(self):
        self.assertEqual(self.config["track_validation"], {"exclusion_radius_px": 2.0})
        self.assertEqual(validate_config(self.config), self.config)
        self.assertEqual(self.config["center_motion"]["rotation_model"], "framewise")
        self.assertEqual(self.config["center_motion"]["radius_fraction"], .0001)

    def test_invalid_prior_combinations_fail_before_running(self):
        for change in ({"track_validation": None}, {"center_radius": 0}, {"frames": 3},
                       {"center_motion": {"model": "rgb_window", "radius_fraction": -1}},
                       {"center_motion": {"model": "rgb_window", "radius_fraction": float("nan")}},
                       {"center_motion": {"model": "rgb_window", "rotation_model": "fixed"}},
                       {"center_motion": {"model": "rgb_window", "unknown": True}}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_config({**self.config, **change})

    def test_other_scenes_do_not_enable_rgb_window(self):
        for name in ("dyn-neu3d", "dyn-4dv"):
            for level in ("fast", "balanced", "quality"):
                config = load_config(ROOT / f"configs/{name}.json", level=level)
                self.assertIsNone(config["center_motion"])
                self.assertFalse(config.get("track_validation"))

    def test_motion_prior_is_opt_in_and_independent_of_the_scene_label(self):
        config = dict(self.config)
        config.pop("center_motion")
        config.pop("track_validation")
        self.assertIsNone(validate_config(config)["center_motion"])
        self.assertEqual(validate_config(config), validate_config({**config, "center_motion": None}))
        renamed = validate_config({**self.config, "scene": "another_capture", "dataset": "custom"})
        self.assertEqual(renamed["center_motion"], self.config["center_motion"])
        self.assertEqual(renamed["track_validation"], self.config["track_validation"])

    def test_rgb_refinement_is_confined_to_quality_presets(self):
        for name in ("huahua", "dyn-dna-rendering"):
            path = ROOT / f"configs/{name}.json"
            default = load_config(path, frames=30)
            self.assertEqual(default, load_config(path, level="balanced", frames=30))
            for level in ("fast", "balanced"):
                config = load_config(path, level=level, frames=30)
                self.assertIsNone(config["center_motion"])
                self.assertFalse(config.get("track_validation"))
                if name == "dyn-dna-rendering":
                    self.assertEqual(config["center_radius"], 0)
            quality = load_config(path, level="quality", frames=30)
            self.assertIsNone(quality["center_radius"])
            self.assertEqual(quality["center_motion"], self.config["center_motion"])
            self.assertEqual(quality["track_validation"], self.config["track_validation"])
            for key in ("static_views", "dynamic_views", "resolution", "frames", "evaluation"):
                self.assertEqual(default[key], quality[key])

    def test_reconstruction_verifier_imports_without_the_source_package(self):
        with tempfile.TemporaryDirectory() as directory:
            snapshot_verification(Path(directory) / "calibration_verification")
            subprocess.run([sys.executable, "-c",
                "from calibration_verification import verify_calibration; "
                "from calibration_verification.trajectory_verification import verify_rgb_motion; "
                "from calibration_verification._engine.runtime.sparse.track_partition import identity_roles; "
                "import sys; assert 'track4d_calib' not in sys.modules"], cwd=directory, check=True)

    def run_pipeline_fixture(self, root, fail=False, config=None):
        code = root / "code"
        (code / "third_party").mkdir(parents=True)
        stages = []
        config = self.config if config is None else config
        target = root / "calibration"

        def execute(command, **kwargs):
            stage = Path(kwargs["stdout"].name).stem
            stages.append(stage)
            if stage == "geometry":
                expected = config["frames"] * len(config["dynamic_views"])
                (target / "PNP_SUMMARY.json").write_text(json.dumps({
                    "success": expected, "expected": expected}))
            if stage in ("closure", "anchor_crop_refinement", "center_motion"):
                folder = {"closure": "poses", "anchor_crop_refinement": "crop_poses",
                          "center_motion": "center_motion/poses"}[stage]
                destination = target / folder / "f000000/cameras_pred"
                destination.mkdir(parents=True)
                (destination / "extri.yml").write_text(stage)
            if stage == "center_motion":
                self.assertTrue(command[1].endswith("refine_rgb_motion.py"))
                self.assertEqual(command[2:], ["--calibration", str(target)])
                if fail:
                    return type("Completed", (), {"returncode": 1})()
                (target / "motion_poses").symlink_to("center_motion/poses", target_is_directory=True)
            return type("Completed", (), {"returncode": 0})()

        args = ["run_pipeline.py", "--root", str(target), "--source", str(root / "source"),
                "--config-json", json.dumps(config), "--dependencies-json",
                json.dumps({"romav2_src": "source", "romav2_weights": "weights"}),
                "--stage-code", str(code), "--gpu", "0"]
        with patch.object(sys, "argv", args), patch("subprocess.run", side_effect=execute), \
                patch("os.sched_setaffinity"):
            runpy.run_path(str(ROOT / "track4d_calib/_engine/runtime/sparse/run_pipeline.py"), run_name="__main__")
        return target, stages

    def test_stage_order_publication_and_timing_include_motion(self):
        with tempfile.TemporaryDirectory() as directory:
            target, stages = self.run_pipeline_fixture(Path(directory))
            self.assertLess(stages.index("track_partition"), stages.index("geometry"))
            self.assertLess(stages.index("anchor_crop_refinement"), stages.index("center_motion"))
            self.assertEqual((target / "poses/f000000/cameras_pred/extri.yml").read_text(), "center_motion")
            self.assertEqual((target / "pre_motion_poses/f000000/cameras_pred/extri.yml").read_text(), "anchor_crop_refinement")
            result = json.loads((target / "RESULT.json").read_text())
            self.assertEqual(result["stages"][-1]["stage"], "center_motion")
            self.assertGreaterEqual(result["wall_seconds"], sum(row["seconds"] for row in result["stages"]))

    def test_failed_motion_stage_does_not_publish_a_complete_result(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(RuntimeError, "center_motion"):
                self.run_pipeline_fixture(root, fail=True)
            self.assertFalse((root / "calibration/RESULT.json").exists())
            self.assertEqual((root / "calibration/poses/f000000/cameras_pred/extri.yml").read_text(), "anchor_crop_refinement")

    def test_disabled_motion_preserves_the_registration_path(self):
        for name in ("huahua", "dyn-dna-rendering", "dyn-neu3d", "dyn-4dv"):
            config = load_config(ROOT / f"configs/{name}.json", frames=30)
            with self.subTest(preset=name), tempfile.TemporaryDirectory() as directory:
                target, stages = self.run_pipeline_fixture(Path(directory), config=config)
                self.assertNotIn("track_partition", stages)
                self.assertNotIn("center_motion", stages)
                self.assertFalse((target / "pre_motion_poses").exists())
                self.assertEqual((target / "poses/f000000/cameras_pred/extri.yml").read_text(), "anchor_crop_refinement")


if __name__ == "__main__":
    unittest.main()
