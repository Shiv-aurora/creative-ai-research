import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from creativeai.cli import main


class CLIPipelineTest(unittest.TestCase):
    def _run_cli(self, argv: list[str]) -> int:
        # Silence command JSON output in tests.
        stream = StringIO()
        with redirect_stdout(stream):
            return main(argv)

    def test_end_to_end_commands(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs_dir = root / "runs"
            scores_dir = root / "scores"
            analysis_dir = root / "analysis"

            rc = self._run_cli(
                [
                    "generate",
                    "--task",
                    "cdat",
                    "--method",
                    "restlessness_loop",
                    "--model",
                    "gemma-2-2b",
                    "--backend",
                    "mock",
                    "--seed",
                    "11",
                    "--temperature",
                    "0.7",
                    "--cue",
                    "forest",
                    "--output-dir",
                    str(runs_dir),
                ]
            )
            self.assertEqual(rc, 0)
            self.assertTrue((runs_dir / "runs.jsonl").exists())

            rc = self._run_cli(["score", "--input", str(runs_dir / "runs.jsonl"), "--output-dir", str(scores_dir)])
            self.assertEqual(rc, 0)
            self.assertTrue((scores_dir / "scores.jsonl").exists())

            rc = self._run_cli(["analyze-frontier", "--runs", str(scores_dir / "scores.jsonl"), "--output-dir", str(analysis_dir)])
            self.assertEqual(rc, 0)
            self.assertTrue((analysis_dir / "frontier_analysis.json").exists())

            rc = self._run_cli(["audit-homogeneity", "--runs", str(runs_dir / "runs.jsonl"), "--output-dir", str(analysis_dir)])
            self.assertEqual(rc, 0)
            self.assertTrue((analysis_dir / "homogeneity_audit.json").exists())

            with (analysis_dir / "frontier_analysis.json").open("r", encoding="utf-8") as f:
                payload = json.load(f)
            self.assertIn("frontier_points", payload)

    def test_score_rejects_mixed_sessions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs_path = root / "runs.jsonl"
            runs = [
                {
                    "run_id": "r1",
                    "session_id": "sess-a",
                    "task_id": "dat",
                    "method": "one_shot",
                    "model_id": "m",
                    "output": ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa"],
                    "validity_flags": {"valid": True, "json_valid": True},
                    "metadata": {"session_id": "sess-a"},
                },
                {
                    "run_id": "r2",
                    "session_id": "sess-b",
                    "task_id": "dat",
                    "method": "one_shot",
                    "model_id": "m",
                    "output": ["lambda", "mu", "nu", "xi", "omicron", "pi", "rho", "sigma", "tau", "upsilon"],
                    "validity_flags": {"valid": True, "json_valid": True},
                    "metadata": {"session_id": "sess-b"},
                },
            ]
            with runs_path.open("w", encoding="utf-8") as f:
                for row in runs:
                    f.write(json.dumps(row) + "\n")

            rc = self._run_cli(["score", "--input", str(runs_path), "--output-dir", str(root / "scores")])
            self.assertEqual(rc, 1)

    def test_generate_grid_quarantines_bad_cell(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs_dir = root / "runs"

            class FakeRecord:
                def __init__(self) -> None:
                    self.json_valid = False
                    self.validity_flags = {"valid": False, "json_valid": False, "problems": ["non_json_output"]}

            def fake_generate_run(*args, **kwargs):  # type: ignore[no-untyped-def]
                return FakeRecord()

            argv = [
                "generate-grid",
                "--tasks",
                "cdat",
                "--methods",
                "one_shot",
                "--models",
                "gemma-2-2b",
                "--backend",
                "mock",
                "--temperatures",
                "0.7",
                "--seeds",
                "11",
                "--limit-cues",
                "25",
                "--health-window",
                "20",
                "--health-min-samples",
                "20",
                "--health-min-json",
                "0.95",
                "--health-min-valid",
                "0.90",
                "--health-action",
                "quarantine_cell",
                "--output-dir",
                str(runs_dir),
            ]
            with patch("creativeai.cli.generate_run", side_effect=fake_generate_run):
                rc = self._run_cli(argv)
            self.assertEqual(rc, 0)

            events_path = runs_dir / "health_events.jsonl"
            self.assertTrue(events_path.exists())
            with events_path.open("r", encoding="utf-8") as f:
                events = [json.loads(line) for line in f if line.strip()]
            self.assertTrue(any(e.get("event_type") == "cell_quarantined" for e in events))
            self.assertTrue(any(int(e.get("skipped_estimate", 0)) > 0 for e in events))

    def test_generate_grid_healthy_cell_has_no_quarantine_events(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs_dir = root / "runs"

            class FakeRecord:
                def __init__(self) -> None:
                    self.json_valid = True
                    self.validity_flags = {"valid": True, "json_valid": True, "problems": []}

            def fake_generate_run(*args, **kwargs):  # type: ignore[no-untyped-def]
                return FakeRecord()

            argv = [
                "generate-grid",
                "--tasks",
                "cdat",
                "--methods",
                "one_shot",
                "--models",
                "gemma-2-2b",
                "--backend",
                "mock",
                "--temperatures",
                "0.7",
                "--seeds",
                "11",
                "--limit-cues",
                "25",
                "--health-window",
                "20",
                "--health-min-samples",
                "20",
                "--health-min-json",
                "0.95",
                "--health-min-valid",
                "0.90",
                "--health-action",
                "quarantine_cell",
                "--output-dir",
                str(runs_dir),
            ]
            with patch("creativeai.cli.generate_run", side_effect=fake_generate_run):
                rc = self._run_cli(argv)
            self.assertEqual(rc, 0)

            events_path = runs_dir / "health_events.jsonl"
            self.assertFalse(events_path.exists())

    def test_generate_grid_dat_repeats_controls_run_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs_dir = root / "runs"
            call_count = {"n": 0}

            class FakeRecord:
                def __init__(self) -> None:
                    self.json_valid = True
                    self.validity_flags = {"valid": True, "json_valid": True, "problems": []}
                    self.tokens_total = 10

            def fake_generate_run(*args, **kwargs):  # type: ignore[no-untyped-def]
                call_count["n"] += 1
                return FakeRecord()

            argv = [
                "generate-grid",
                "--tasks",
                "dat",
                "--methods",
                "one_shot",
                "--models",
                "gemma-2-2b",
                "--backend",
                "mock",
                "--temperatures",
                "0.7",
                "--seeds",
                "11",
                "--dat-repeats",
                "4",
                "--health-min-samples",
                "100",
                "--output-dir",
                str(runs_dir),
            ]
            with patch("creativeai.cli.generate_run", side_effect=fake_generate_run):
                rc = self._run_cli(argv)
            self.assertEqual(rc, 0)
            self.assertEqual(call_count["n"], 4)


if __name__ == "__main__":
    unittest.main()
