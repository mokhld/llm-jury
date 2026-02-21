from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import typer

from llm_jury.cli.main import main


class CLITests(unittest.TestCase):
    def test_classify_writes_output_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            input_path = Path(tmp) / "input.jsonl"
            output_path = Path(tmp) / "output.jsonl"
            input_path.write_text(
                "\n".join(
                    [
                        json.dumps({"text": "a", "predicted_label": "safe", "predicted_confidence": 0.95}),
                        json.dumps({"text": "b", "predicted_label": "unsafe", "predicted_confidence": 0.96}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            main(
                [
                    "classify",
                    "--input",
                    str(input_path),
                    "--output",
                    str(output_path),
                    "--classifier",
                    "function",
                    "--personas",
                    "content_moderation",
                    "--judge",
                    "majority",
                    "--labels",
                    "safe,unsafe",
                    "--threshold",
                    "0.7",
                ]
            )
            lines = output_path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 2)
            first = json.loads(lines[0])
            self.assertIn("label", first)
            self.assertIn("was_escalated", first)

    def test_calibrate_prints_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            input_path = Path(tmp) / "calib.jsonl"
            input_path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "text": "t1",
                                "label": "safe",
                                "predicted_label": "safe",
                                "predicted_confidence": 0.9,
                            }
                        ),
                        json.dumps(
                            {
                                "text": "t2",
                                "label": "unsafe",
                                "predicted_label": "unsafe",
                                "predicted_confidence": 0.4,
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            buf = io.StringIO()
            with redirect_stdout(buf):
                main(
                    [
                        "calibrate",
                        "--input",
                        str(input_path),
                        "--classifier",
                        "function",
                        "--personas",
                        "content_moderation",
                        "--judge",
                        "majority",
                        "--labels",
                        "safe,unsafe",
                    ]
                )
            payload = json.loads(buf.getvalue().strip())
            self.assertIn("best_threshold", payload)
            self.assertIn("rows", payload)

    def test_calibrate_requires_ground_truth_label(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            input_path = Path(tmp) / "missing-labels.jsonl"
            input_path.write_text(
                "\n".join(
                    [
                        json.dumps({"text": "t1", "predicted_label": "safe", "predicted_confidence": 0.9}),
                        json.dumps({"text": "t2", "predicted_label": "unsafe", "predicted_confidence": 0.4}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(typer.BadParameter, "ground-truth 'label'"):
                main(
                    [
                        "calibrate",
                        "--input",
                        str(input_path),
                        "--classifier",
                        "function",
                        "--personas",
                        "content_moderation",
                        "--judge",
                        "majority",
                        "--labels",
                        "safe,unsafe",
                    ]
                )


if __name__ == "__main__":
    unittest.main()
