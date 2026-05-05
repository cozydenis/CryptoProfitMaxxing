"""Tests for PyTorch LSTM classifier — CPU only, synthetic data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from src.config import FEATURE_COLUMNS, TARGET_COLUMN
from src.models.lstm import LSTMClassifier, create_sequences

# ---------------------------------------------------------------------------
# create_sequences
# ---------------------------------------------------------------------------


class TestCreateSequences:
    def test_output_shape(self) -> None:
        X = np.random.randn(20, 5)
        y = np.random.randint(0, 2, size=20)
        X_seq, y_seq = create_sequences(X, y, seq_len=5)
        assert X_seq.shape == (16, 5, 5)  # 20 - 5 + 1 = 16
        assert y_seq.shape == (16,)

    def test_content_matches_source(self) -> None:
        X = np.arange(30).reshape(10, 3).astype(float)
        y = np.arange(10)
        X_seq, y_seq = create_sequences(X, y, seq_len=3)
        # First window should be rows 0-2, label from row 2
        np.testing.assert_array_equal(X_seq[0], X[0:3])
        assert y_seq[0] == y[2]
        # Last window should be rows 7-9, label from row 9
        np.testing.assert_array_equal(X_seq[-1], X[7:10])
        assert y_seq[-1] == y[9]

    def test_seq_len_one(self) -> None:
        X = np.random.randn(5, 2)
        y = np.arange(5)
        X_seq, y_seq = create_sequences(X, y, seq_len=1)
        assert X_seq.shape == (5, 1, 2)
        assert y_seq.shape == (5,)

    def test_too_short_raises(self) -> None:
        X = np.random.randn(3, 2)
        y = np.arange(3)
        with pytest.raises(ValueError, match="at least"):
            create_sequences(X, y, seq_len=5)


# ---------------------------------------------------------------------------
# LSTMClassifier
# ---------------------------------------------------------------------------


@pytest.fixture
def small_classifier() -> LSTMClassifier:
    """Tiny LSTM for fast CPU tests."""
    return LSTMClassifier(
        seq_len=3, hidden_size=8, num_layers=1,
        epochs=3, batch_size=16, random_state=42,
    )


@pytest.fixture
def synthetic_xy() -> tuple[pd.DataFrame, pd.Series]:
    """100-row synthetic dataset with 11 features matching FEATURE_COLUMNS."""
    rng = np.random.default_rng(42)
    X = pd.DataFrame(
        rng.standard_normal((100, len(FEATURE_COLUMNS))),
        columns=FEATURE_COLUMNS,
    )
    y = pd.Series(rng.integers(0, 2, size=100), name=TARGET_COLUMN)
    return X, y


class TestLSTMClassifier:
    def test_fit_returns_self(
        self, small_classifier: LSTMClassifier, synthetic_xy: tuple
    ) -> None:
        X, y = synthetic_xy
        result = small_classifier.fit(X, y)
        assert result is small_classifier

    def test_predict_shape_and_values(
        self, small_classifier: LSTMClassifier, synthetic_xy: tuple
    ) -> None:
        X, y = synthetic_xy
        small_classifier.fit(X, y)
        preds = small_classifier.predict(X)
        assert preds.shape == (len(X),)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_predict_proba_shape(
        self, small_classifier: LSTMClassifier, synthetic_xy: tuple
    ) -> None:
        X, y = synthetic_xy
        small_classifier.fit(X, y)
        proba = small_classifier.predict_proba(X)
        assert proba.shape == (len(X), 2)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_single_row_predict(
        self, small_classifier: LSTMClassifier, synthetic_xy: tuple
    ) -> None:
        X, y = synthetic_xy
        small_classifier.fit(X, y)
        single = X.iloc[[-1]]
        pred = small_classifier.predict(single)
        assert pred.shape == (1,)
        assert pred[0] in (0, 1)

    def test_evaluate_compatible(
        self, small_classifier: LSTMClassifier, synthetic_xy: tuple
    ) -> None:
        """evaluate() from baseline.py must work with LSTMClassifier."""
        from src.models.baseline import evaluate

        X, y = synthetic_xy
        small_classifier.fit(X, y)
        metrics = evaluate(small_classifier, X, y)
        assert "accuracy" in metrics
        assert "f1" in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0

    def test_get_params(self, small_classifier: LSTMClassifier) -> None:
        params = small_classifier.get_params()
        assert params["seq_len"] == 3
        assert params["hidden_size"] == 8
        assert params["epochs"] == 3

    def test_reproducibility(self, synthetic_xy: tuple) -> None:
        X, y = synthetic_xy
        clf1 = LSTMClassifier(seq_len=3, hidden_size=8, epochs=2, random_state=99)
        clf2 = LSTMClassifier(seq_len=3, hidden_size=8, epochs=2, random_state=99)
        clf1.fit(X, y)
        clf2.fit(X, y)
        np.testing.assert_array_equal(clf1.predict(X), clf2.predict(X))
