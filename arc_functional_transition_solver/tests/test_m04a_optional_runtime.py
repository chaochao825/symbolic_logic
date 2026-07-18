from __future__ import annotations

import ast
import subprocess
import sys
import unittest
from pathlib import Path


class M04AOptionalRuntimeTests(unittest.TestCase):
    def test_default_m04a_modules_do_not_import_torch(self) -> None:
        self.assertNotIn("torch", sys.modules)
        __import__("afts_arc.m04a_contract")
        __import__("afts_arc.m04a_campaign_command")
        __import__("afts_arc.m04a_data")
        __import__("afts_arc.m04a_evidence")
        __import__("afts_arc.m04a_pool_staging")
        __import__("afts_arc.m04a_production_data_contract")
        __import__("afts_arc.m04a_rearc_index")
        __import__("afts_arc.m04a_split_privileged")
        __import__("afts_arc.m04a_train_contract")
        __import__("afts_arc.m04a_validation_manifest")
        self.assertNotIn("torch", sys.modules)

    def test_torch_modules_are_explicit_and_not_default_discovery(self) -> None:
        project = Path(__file__).resolve().parents[1]
        runtime = project / "src" / "afts_arc" / "m04a_torch_runtime.py"
        model = project / "src" / "afts_arc" / "m04a_model.py"
        sampler = project / "src" / "afts_arc" / "m04a_sample.py"
        trainer = project / "src" / "afts_arc" / "m04a_train.py"
        campaign = project / "src" / "afts_arc" / "m04a_campaign.py"
        preflight = project / "src" / "afts_arc" / "m04a_preflight.py"
        neural_suite = project / "tests" / "m04a_torch_suite.py"
        sampler_suite = project / "tests" / "m04a_sample_torch_suite.py"
        trainer_suite = project / "tests" / "m04a_train_torch_suite.py"
        campaign_suite = project / "tests" / "m04a_campaign_torch_suite.py"
        preflight_suite = project / "tests" / "m04a_preflight_torch_suite.py"
        for path in (
            runtime,
            model,
            sampler,
            trainer,
            campaign,
            preflight,
            neural_suite,
            sampler_suite,
            trainer_suite,
            campaign_suite,
            preflight_suite,
        ):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            self.assertIsInstance(tree, ast.Module)
        self.assertFalse(neural_suite.name.startswith("test_"))
        self.assertFalse(sampler_suite.name.startswith("test_"))
        self.assertFalse(trainer_suite.name.startswith("test_"))
        self.assertFalse(campaign_suite.name.startswith("test_"))
        self.assertFalse(preflight_suite.name.startswith("test_"))

    def test_neural_launcher_rejects_unreviewed_command_before_torch_import(self) -> None:
        project = Path(__file__).resolve().parents[1]
        launcher = project / "scripts" / "afts_arc_m04a.py"
        completed = subprocess.run(
            [sys.executable, str(launcher), "audit-data"],
            cwd=project,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("reviewed M04a neural command", completed.stderr)


if __name__ == "__main__":
    unittest.main()
