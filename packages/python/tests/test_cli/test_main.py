from __future__ import annotations

import asyncio
import io
import json
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import typer

from llm_jury.cli.main import _build_classifier, _check_debate_mode, main
from tests.helpers import FakeLLMClient


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


class CLITests(unittest.TestCase):
    def test_classify_writes_output_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            input_path = Path(tmp) / "input.jsonl"
            output_path = Path(tmp) / "output.jsonl"
            input_path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "text": "a",
                                "predicted_label": "safe",
                                "predicted_confidence": 0.95,
                            }
                        ),
                        json.dumps(
                            {
                                "text": "b",
                                "predicted_label": "unsafe",
                                "predicted_confidence": 0.96,
                            }
                        ),
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
            # Output rows are Verdict.to_dict(); the TS CLI writes the same keys.
            for key in (
                "debate_degraded",
                "persona_failures",
                "library_version",
                "created_at",
                "primary_result",
                "total_cost_usd",
            ):
                self.assertIn(key, first)

    def test_classify_records_per_row_errors_without_losing_batch(self) -> None:
        from unittest.mock import patch

        from llm_jury.jury.core import Jury

        original_classify = Jury.classify

        async def flaky_classify(self: Jury, text: str):
            if text == "boom":
                raise RuntimeError("row failed")
            return await original_classify(self, text)

        with tempfile.TemporaryDirectory() as tmp:
            input_path = Path(tmp) / "input.jsonl"
            output_path = Path(tmp) / "output.jsonl"
            input_path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "text": "boom",
                                "predicted_label": "safe",
                                "predicted_confidence": 0.95,
                            }
                        ),
                        json.dumps(
                            {
                                "text": "fine",
                                "predicted_label": "safe",
                                "predicted_confidence": 0.95,
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            with patch.object(Jury, "classify", flaky_classify):
                main(
                    [
                        "classify",
                        "--input",
                        str(input_path),
                        "--output",
                        str(output_path),
                        "--classifier",
                        "function",
                        "--judge",
                        "majority",
                        "--labels",
                        "safe,unsafe",
                    ]
                )

            lines = output_path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 2)
            error_row = json.loads(lines[0])
            ok_row = json.loads(lines[1])
            self.assertIn("RuntimeError", error_row["error"])
            self.assertEqual(error_row["text"], "boom")
            self.assertEqual(ok_row["label"], "safe")

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
                        json.dumps(
                            {
                                "text": "t1",
                                "predicted_label": "safe",
                                "predicted_confidence": 0.9,
                            }
                        ),
                        json.dumps(
                            {
                                "text": "t2",
                                "predicted_label": "unsafe",
                                "predicted_confidence": 0.4,
                            }
                        ),
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


CLASSIFY_BASE = ["--classifier", "function", "--judge", "majority"]


class FunctionSpecTests(unittest.TestCase):
    """The 'function' spec replays stored predictions and never reads ground truth."""

    def test_calibrate_rejects_rows_without_predictions(self) -> None:
        # Rows with only text + ground-truth label used to be scored against
        # themselves: accuracy 1.0 at every threshold.
        with tempfile.TemporaryDirectory() as tmp:
            input_path = Path(tmp) / "calib.jsonl"
            _write_rows(
                input_path,
                [{"text": "t1", "label": "safe"}, {"text": "t2", "label": "unsafe"}],
            )
            with self.assertRaisesRegex(
                typer.BadParameter, r"predicted_label.*2 row\(s\): 1, 2\."
            ):
                main(["calibrate", "--input", str(input_path), *CLASSIFY_BASE])

    def test_classify_rejects_rows_without_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            input_path = Path(tmp) / "input.jsonl"
            output_path = Path(tmp) / "output.jsonl"
            _write_rows(
                input_path,
                [
                    {
                        "text": "a",
                        "predicted_label": "safe",
                        "predicted_confidence": 0.9,
                    },
                    {"text": "b", "label": "unsafe"},
                ],
            )
            with self.assertRaisesRegex(typer.BadParameter, r"1 row\(s\): 2\."):
                main(
                    [
                        "classify",
                        "--input",
                        str(input_path),
                        "--output",
                        str(output_path),
                        *CLASSIFY_BASE,
                    ]
                )
            self.assertFalse(output_path.exists())

    def test_predictions_come_from_predicted_fields_not_ground_truth(self) -> None:
        rows = [
            {
                "text": "t1",
                "label": "safe",
                "predicted_label": "unsafe",
                "predicted_confidence": 0.8,
            },
            {
                "text": "t2",
                "label": "unsafe",
                "predicted_label": "unsafe",
                "predicted_confidence": "0.6",
            },
        ]
        clf, is_mock = _build_classifier("function", ["safe", "unsafe"], rows)
        self.assertTrue(is_mock)
        first = asyncio.run(clf.classify("t1"))
        second = asyncio.run(clf.classify("t2"))
        self.assertEqual((first.label, first.confidence), ("unsafe", 0.8))
        self.assertEqual((second.label, second.confidence), ("unsafe", 0.6))

    def test_error_names_at_most_five_rows(self) -> None:
        rows = [{"text": f"t{i}", "label": "safe"} for i in range(8)]
        with self.assertRaisesRegex(
            typer.BadParameter, r"8 row\(s\): 1, 2, 3, 4, 5 \(and 3 more\)\."
        ):
            _build_classifier("function", ["safe", "unsafe"], rows)

    def test_invalid_predicted_values_are_rejected(self) -> None:
        bad_rows = [
            {"predicted_label": "safe", "predicted_confidence": 0.9},  # no text
            {"text": "x", "predicted_label": None, "predicted_confidence": 0.9},
            {"text": "x", "predicted_label": "  ", "predicted_confidence": 0.9},
            {"text": "x", "predicted_label": "safe", "predicted_confidence": None},
            {"text": "x", "predicted_label": "safe", "predicted_confidence": "high"},
            {"text": "x", "predicted_label": "safe", "predicted_confidence": 1.5},
            {"text": "x", "predicted_label": "safe", "predicted_confidence": -0.1},
            {"text": "x", "predicted_label": "safe", "predicted_confidence": True},
            {"text": "x", "predicted_label": "safe", "predicted_confidence": "nan"},
            {"text": "x", "predicted_label": "safe", "predicted_confidence": [0.9]},
        ]
        for row in bad_rows:
            with self.subTest(row=row):
                with self.assertRaisesRegex(typer.BadParameter, "1 row"):
                    _build_classifier("function", ["safe", "unsafe"], [row])

    def test_duplicate_text_with_same_prediction_is_allowed(self) -> None:
        row = {"text": "dup", "predicted_label": "safe", "predicted_confidence": 0.9}
        clf, _ = _build_classifier("function", ["safe", "unsafe"], [row, dict(row)])
        self.assertEqual(asyncio.run(clf.classify("dup")).label, "safe")

    def test_duplicate_text_with_different_predictions_is_rejected(self) -> None:
        rows = [
            {"text": "dup", "predicted_label": "safe", "predicted_confidence": 0.9},
            {"text": "other", "predicted_label": "safe", "predicted_confidence": 0.9},
            {"text": "dup", "predicted_label": "unsafe", "predicted_confidence": 0.9},
        ]
        with self.assertRaisesRegex(
            typer.BadParameter, r"repeat an earlier text.*: 3\."
        ):
            _build_classifier("function", ["safe", "unsafe"], rows)


class HuggingFaceSpecTests(unittest.TestCase):
    def setUp(self) -> None:
        fake = types.ModuleType("transformers")
        fake.pipeline = lambda *args, **kwargs: (lambda text: [[]])
        sys.modules["transformers"] = fake

    def tearDown(self) -> None:
        sys.modules.pop("transformers", None)
        sys.modules.pop("llm_jury.classifiers.huggingface_adapter", None)

    def test_labels_flag_is_passed_to_huggingface_classifier(self) -> None:
        clf, is_mock = _build_classifier(
            "huggingface:some/model",
            ["safe", "unsafe"],
            [],
            explicit_labels=["safe", "unsafe"],
        )
        self.assertFalse(is_mock)
        self.assertEqual(clf.labels, ["safe", "unsafe"])

    def test_without_labels_flag_labels_come_from_the_model(self) -> None:
        clf, _ = _build_classifier("huggingface:some/model", ["safe", "unsafe"], [])
        self.assertEqual(clf.labels, [])


class OptionValidationTests(unittest.TestCase):
    def _classify_args(self, tmp: str) -> list[str]:
        input_path = Path(tmp) / "input.jsonl"
        _write_rows(
            input_path,
            [{"text": "a", "predicted_label": "safe", "predicted_confidence": 0.9}],
        )
        return [
            "classify",
            "--input",
            str(input_path),
            "--output",
            str(Path(tmp) / "output.jsonl"),
            *CLASSIFY_BASE,
        ]

    def _calibrate_args(self, tmp: str) -> list[str]:
        input_path = Path(tmp) / "calib.jsonl"
        _write_rows(
            input_path,
            [
                {
                    "text": "a",
                    "label": "safe",
                    "predicted_label": "safe",
                    "predicted_confidence": 0.9,
                }
            ],
        )
        return ["calibrate", "--input", str(input_path), *CLASSIFY_BASE]

    def test_classify_rejects_out_of_range_numbers(self) -> None:
        cases = [
            ("--threshold", "1.5"),
            ("--threshold", "-0.1"),
            ("--threshold", "nan"),
            ("--concurrency", "0"),
            ("--debate-concurrency", "0"),
            ("--max-rounds", "0"),
            ("--max-debate-cost", "-1"),
            ("--max-debate-cost", "inf"),
        ]
        for flag, value in cases:
            with self.subTest(flag=flag, value=value):
                with tempfile.TemporaryDirectory() as tmp:
                    with self.assertRaisesRegex(typer.BadParameter, "must be"):
                        main([*self._classify_args(tmp), flag, value])

    def test_calibrate_rejects_out_of_range_numbers(self) -> None:
        cases = [
            ("--initial-threshold", "2"),
            ("--error-cost", "-1"),
            ("--escalation-cost", "inf"),
            ("--max-rounds", "0"),
            ("--debate-concurrency", "0"),
        ]
        for flag, value in cases:
            with self.subTest(flag=flag, value=value):
                with tempfile.TemporaryDirectory() as tmp:
                    with self.assertRaisesRegex(typer.BadParameter, "must be"):
                        main([*self._calibrate_args(tmp), flag, value])

    def test_unknown_debate_mode_lists_valid_modes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(
                typer.BadParameter,
                "independent, sequential, deliberation, adversarial",
            ):
                main([*self._classify_args(tmp), "--debate-mode", "chaos"])

    def test_debate_mode_is_case_insensitive(self) -> None:
        self.assertEqual(_check_debate_mode(" Deliberation "), "deliberation")


EVAL_ROWS = [
    # Confident and right: never debated at --band-upper 0.95.
    {
        "text": "t1",
        "label": "safe",
        "predicted_label": "safe",
        "predicted_confidence": 0.99,
    },
    # Unsure and wrong: the fake personas answer "safe" and fix it.
    {
        "text": "t2",
        "label": "safe",
        "predicted_label": "unsafe",
        "predicted_confidence": 0.6,
    },
    {
        "text": "t3",
        "label": "unsafe",
        "predicted_label": "unsafe",
        "predicted_confidence": 0.97,
    },
]


class EvalCommandTests(unittest.TestCase):
    """`llm-jury eval` end to end with an injected fake LLM client."""

    def _run(
        self, tmp: str, *extra: str, client: FakeLLMClient | None = None
    ) -> tuple[dict, FakeLLMClient]:
        input_path = Path(tmp) / "labelled.jsonl"
        _write_rows(input_path, EVAL_ROWS)
        client = client or FakeLLMClient()
        buf = io.StringIO()
        with redirect_stdout(buf):
            main(
                ["eval", "--input", str(input_path), "--labels", "safe,unsafe", *extra],
                llm_client=client,
            )
        return json.loads(buf.getvalue().strip()), client

    def test_eval_prints_summary_sweep_and_best_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "report.json"
            payload, client = self._run(
                tmp,
                "--judge",
                "majority",
                "--thresholds",
                "0.5,0.7,0.9",
                "--output",
                str(output_path),
            )

            summary = payload["summary"]
            self.assertEqual(summary["n"], 3)
            self.assertEqual(summary["debated"], 1)
            self.assertEqual(summary["flips_helped"], 1)
            self.assertEqual(summary["flips_hurt"], 0)
            self.assertAlmostEqual(summary["debate_cost_usd"], 0.003)
            self.assertEqual(summary["unpriced_calls"], 0)
            self.assertEqual(
                [row["threshold"] for row in payload["sweep"]], [0.5, 0.7, 0.9]
            )
            self.assertEqual(payload["best_threshold"], 0.7)
            # Three content_moderation personas, one independent round, no LLM judge.
            self.assertEqual(len(client.calls), 3)

            full = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(full["best_threshold"], 0.7)
            self.assertEqual(full["band_upper"], 0.95)
            self.assertEqual(len(full["items"]), 3)
            self.assertEqual(full["items"][1]["jury_label"], "safe")
            self.assertFalse(full["items"][0]["debated"])

    def test_eval_uses_the_injected_client_for_the_llm_judge(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            payload, client = self._run(tmp)
        self.assertEqual(payload["summary"]["fallbacks"], {})
        self.assertEqual(len(client.calls), 4)  # 3 personas + the judge

    def test_max_escalations_stops_before_any_llm_call(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            client = FakeLLMClient()
            with self.assertRaisesRegex(typer.BadParameter, "max_escalations=0"):
                self._run(tmp, "--max-escalations", "0", client=client)
            self.assertEqual(client.calls, [])

    def test_thresholds_above_band_upper_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            client = FakeLLMClient()
            with self.assertRaisesRegex(typer.BadParameter, "above --band-upper"):
                self._run(
                    tmp, "--band-upper", "0.8", "--thresholds", "0.9", client=client
                )
            self.assertEqual(client.calls, [])

    def test_invalid_eval_numbers_are_rejected(self) -> None:
        cases = [
            ("--band-upper", "1.5"),
            ("--max-escalations", "-1"),
            ("--thresholds", "0.5,abc"),
            ("--error-cost", "-1"),
            ("--escalation-cost", "nan"),
            ("--concurrency", "0"),
        ]
        for flag, value in cases:
            with self.subTest(flag=flag, value=value):
                with tempfile.TemporaryDirectory() as tmp:
                    with self.assertRaisesRegex(typer.BadParameter, "must be"):
                        self._run(tmp, flag, value)


class CalibrateUseJuryTests(unittest.TestCase):
    def _calibrate(
        self, tmp: str, *extra: str, client: FakeLLMClient | None = None
    ) -> tuple[dict, str, FakeLLMClient]:
        input_path = Path(tmp) / "calib.jsonl"
        _write_rows(input_path, EVAL_ROWS)
        client = client or FakeLLMClient()
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            main(
                [
                    "calibrate",
                    "--input",
                    str(input_path),
                    "--labels",
                    "safe,unsafe",
                    *extra,
                ],
                llm_client=client,
            )
        return json.loads(out.getvalue().strip()), err.getvalue(), client

    def test_use_jury_measures_the_jury(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report, err, client = self._calibrate(
                tmp, "--use-jury", "--judge", "majority"
            )
        self.assertTrue(report["use_jury"])
        self.assertEqual(report["summary"]["debated"], 1)
        self.assertEqual(report["summary"]["flips_helped"], 1)
        for row in report["rows"]:
            self.assertIn("system_accuracy", row)
            self.assertIn("jury_accuracy", row)
        # t2 (0.6) is wrong unless escalated, so the best threshold escalates it.
        self.assertGreater(report["best_threshold"], 0.6)
        self.assertEqual(len(client.calls), 3)
        self.assertEqual(err, "")

    def test_cheap_mode_says_jury_flags_are_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report, err, client = self._calibrate(
                tmp, "--judge", "majority", "--max-rounds", "2"
            )
        self.assertFalse(report["use_jury"])
        self.assertIn("--judge, --max-rounds", err)
        self.assertIn("--use-jury", err)
        self.assertEqual(client.calls, [])

    def test_cheap_mode_without_jury_flags_prints_no_note(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _, err, _ = self._calibrate(tmp)
        self.assertEqual(err, "")


if __name__ == "__main__":
    unittest.main()
