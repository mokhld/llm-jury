from __future__ import annotations

import sys
import types
import unittest


def _install_fake_transformers(pipeline_factory) -> None:
    fake = types.ModuleType("transformers")
    fake.pipeline = pipeline_factory
    sys.modules["transformers"] = fake


def _uninstall_fake_transformers() -> None:
    sys.modules.pop("transformers", None)
    # Force a fresh import of the adapter so each test re-binds `pipeline`.
    sys.modules.pop("llm_jury.classifiers.huggingface_adapter", None)


class HuggingFaceAdapterTests(unittest.IsolatedAsyncioTestCase):
    def tearDown(self) -> None:
        _uninstall_fake_transformers()

    async def test_returns_top_scoring_label(self) -> None:
        captured_kwargs: dict[str, object] = {}

        def fake_pipeline(task: str, *, model: str, device: str, top_k):  # noqa: ANN001
            captured_kwargs.update(task=task, model=model, device=device, top_k=top_k)

            def call(text: str):
                # `top_k=None` returns the full distribution, wrapped in a list
                # so caller does `pipe(text)[0]` to get the inner list of dicts.
                return [
                    [
                        {"label": "safe", "score": 0.2},
                        {"label": "unsafe", "score": 0.7},
                        {"label": "spam", "score": 0.1},
                    ]
                ]

            return call

        _install_fake_transformers(fake_pipeline)
        from llm_jury.classifiers.huggingface_adapter import HuggingFaceClassifier

        clf = HuggingFaceClassifier(model_name="distilbert-x", device="cpu")
        result = await clf.classify("input")

        self.assertEqual(result.label, "unsafe")
        self.assertAlmostEqual(result.confidence, 0.7)
        self.assertEqual(captured_kwargs["task"], "text-classification")
        self.assertEqual(captured_kwargs["model"], "distilbert-x")
        self.assertEqual(captured_kwargs["device"], "cpu")
        self.assertIsNone(captured_kwargs["top_k"])

    async def test_auto_populates_labels_when_empty(self) -> None:
        def fake_pipeline(task: str, *, model: str, device: str, top_k):  # noqa: ANN001
            def call(text: str):
                return [
                    [
                        {"label": "POSITIVE", "score": 0.9},
                        {"label": "NEGATIVE", "score": 0.1},
                    ]
                ]

            return call

        _install_fake_transformers(fake_pipeline)
        from llm_jury.classifiers.huggingface_adapter import HuggingFaceClassifier

        clf = HuggingFaceClassifier(model_name="m", labels=None)
        await clf.classify("input")

        # After one call, labels are populated from the result distribution.
        self.assertEqual(clf.labels, ["POSITIVE", "NEGATIVE"])

    async def test_explicit_labels_are_preserved(self) -> None:
        def fake_pipeline(task: str, *, model: str, device: str, top_k):  # noqa: ANN001
            def call(text: str):
                return [
                    [{"label": "RAW_A", "score": 0.6}, {"label": "RAW_B", "score": 0.4}]
                ]

            return call

        _install_fake_transformers(fake_pipeline)
        from llm_jury.classifiers.huggingface_adapter import HuggingFaceClassifier

        clf = HuggingFaceClassifier(model_name="m", labels=["A", "B"])
        await clf.classify("input")

        # Adapter does not overwrite explicit labels — the user-provided list wins.
        self.assertEqual(clf.labels, ["A", "B"])

    async def test_passes_raw_output_through(self) -> None:
        payload = [{"label": "a", "score": 0.55}, {"label": "b", "score": 0.45}]

        def fake_pipeline(task: str, *, model: str, device: str, top_k):  # noqa: ANN001
            def call(text: str):
                return [payload]

            return call

        _install_fake_transformers(fake_pipeline)
        from llm_jury.classifiers.huggingface_adapter import HuggingFaceClassifier

        clf = HuggingFaceClassifier(model_name="m")
        result = await clf.classify("input")
        self.assertEqual(result.raw_output, payload)


if __name__ == "__main__":
    unittest.main()
