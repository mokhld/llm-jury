from __future__ import annotations

import asyncio
import sys
import threading
import types
import unittest


def _install_fake_numpy() -> None:
    """Inject a minimal fake `numpy` for SklearnClassifier, which only needs argmax.

    The adapter does `import numpy as np` inside classify(), so we register the
    module before each test. Tests in this file run without scikit-learn or
    real numpy installed.
    """

    fake = types.ModuleType("numpy")

    def argmax(seq):
        best_idx = 0
        best_val = seq[0]
        for i, value in enumerate(seq):
            if value > best_val:
                best_val = value
                best_idx = i
        return best_idx

    fake.argmax = argmax
    sys.modules["numpy"] = fake


def _uninstall_fake_numpy() -> None:
    sys.modules.pop("numpy", None)


class _FakeModel:
    def __init__(self, prob_rows: list[list[float]]) -> None:
        self.prob_rows = prob_rows
        self.last_features = None

    def predict_proba(self, features):
        self.last_features = features
        return self.prob_rows


class _FakeModelWithClasses(_FakeModel):
    def __init__(self, prob_rows: list[list[float]], classes: list) -> None:
        super().__init__(prob_rows)
        self.classes_ = classes


class _FakeVectorizer:
    def __init__(self) -> None:
        self.last_input = None

    def transform(self, texts):
        self.last_input = texts
        return [[1.0, 0.0]]  # opaque to the adapter


class SklearnAdapterTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        _install_fake_numpy()

    def tearDown(self) -> None:
        _uninstall_fake_numpy()

    async def test_returns_argmax_label_and_confidence(self) -> None:
        from llm_jury.classifiers.sklearn_adapter import SklearnClassifier

        model = _FakeModel([[0.1, 0.7, 0.2]])
        clf = SklearnClassifier(model=model, labels=["safe", "unsafe", "spam"])

        result = await clf.classify("input")

        self.assertEqual(result.label, "unsafe")
        self.assertAlmostEqual(result.confidence, 0.7)
        self.assertEqual(result.raw_output, [0.1, 0.7, 0.2])

    async def test_uses_vectorizer_when_provided(self) -> None:
        from llm_jury.classifiers.sklearn_adapter import SklearnClassifier

        model = _FakeModel([[0.9, 0.1]])
        vectorizer = _FakeVectorizer()
        clf = SklearnClassifier(model=model, labels=["a", "b"], vectorizer=vectorizer)

        await clf.classify("hello")

        # Vectorizer received the input text wrapped in a list (sklearn convention).
        self.assertEqual(vectorizer.last_input, ["hello"])
        # Model receives the vectorizer output, not the raw text.
        self.assertEqual(model.last_features, [[1.0, 0.0]])

    async def test_without_vectorizer_passes_raw_text_list(self) -> None:
        from llm_jury.classifiers.sklearn_adapter import SklearnClassifier

        model = _FakeModel([[0.6, 0.4]])
        clf = SklearnClassifier(model=model, labels=["x", "y"])

        await clf.classify("payload")

        # Without a vectorizer, the adapter passes `[text]` straight to predict_proba.
        self.assertEqual(model.last_features, ["payload"])

    async def test_ties_resolve_to_first_index(self) -> None:
        from llm_jury.classifiers.sklearn_adapter import SklearnClassifier

        model = _FakeModel([[0.5, 0.5]])
        clf = SklearnClassifier(model=model, labels=["first", "second"])

        result = await clf.classify("input")

        # Fake argmax mirrors numpy: strictly-greater wins, so a tie returns the
        # first index. Pinning so a future swap to actual numpy keeps the same
        # tie-break direction.
        self.assertEqual(result.label, "first")

    async def test_reports_zero_cost_and_float_probabilities(self) -> None:
        from llm_jury.classifiers.sklearn_adapter import SklearnClassifier

        clf = SklearnClassifier(model=_FakeModel([[0.25, 0.75]]), labels=["a", "b"])
        result = await clf.classify("input")
        self.assertEqual(result.cost_usd, 0.0)
        self.assertTrue(all(type(p) is float for p in result.raw_output))

    async def test_columns_follow_model_classes_order(self) -> None:
        from llm_jury.classifiers.sklearn_adapter import SklearnClassifier

        # sklearn sorts classes_, so the columns are [safe, unsafe] even though
        # the user listed the labels the other way round.
        model = _FakeModelWithClasses([[0.9, 0.1]], classes=["safe", "unsafe"])
        clf = SklearnClassifier(model=model, labels=["unsafe", "safe"])
        result = await clf.classify("input")
        self.assertEqual(result.label, "safe")
        self.assertAlmostEqual(result.confidence, 0.9)
        self.assertEqual(clf.labels, ["unsafe", "safe"])

    async def test_non_string_classes_map_labels_by_position(self) -> None:
        from llm_jury.classifiers.sklearn_adapter import SklearnClassifier

        model = _FakeModelWithClasses([[0.2, 0.8]], classes=[0, 1])
        clf = SklearnClassifier(model=model, labels=["ham", "spam"])
        result = await clf.classify("input")
        self.assertEqual(result.label, "spam")

    def test_class_count_mismatch_raises_at_construction(self) -> None:
        from llm_jury.classifiers.sklearn_adapter import SklearnClassifier

        model = _FakeModelWithClasses([[0.2, 0.3, 0.5]], classes=["a", "b", "c"])
        with self.assertRaisesRegex(ValueError, "3 entries.*2 labels"):
            SklearnClassifier(model=model, labels=["a", "b"])

    async def test_inference_runs_off_the_event_loop_thread(self) -> None:
        from llm_jury.classifiers.sklearn_adapter import SklearnClassifier

        loop_thread = threading.get_ident()
        call_threads: list[int] = []
        barrier = threading.Barrier(2, timeout=5)

        class _BlockingModel:
            def predict_proba(self, features):
                call_threads.append(threading.get_ident())
                barrier.wait()
                return [[0.3, 0.7]]

        clf = SklearnClassifier(model=_BlockingModel(), labels=["a", "b"])
        results = await asyncio.gather(clf.classify("x"), clf.classify("y"))
        self.assertEqual([r.label for r in results], ["b", "b"])
        self.assertNotIn(loop_thread, call_threads)


if __name__ == "__main__":
    unittest.main()
