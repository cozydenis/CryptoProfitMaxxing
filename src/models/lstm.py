"""PyTorch LSTM binary classifier with sklearn-compatible interface.

The LSTMClassifier wraps a small LSTM network and exposes fit(), predict(),
and predict_proba() so it slots into the existing evaluate() and dashboard
code without modification.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from numpy.typing import NDArray
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


def create_sequences(
    X: NDArray[np.floating],
    y: NDArray[np.integer],
    seq_len: int,
) -> tuple[NDArray[np.floating], NDArray[np.integer]]:
    """Slide a window of *seq_len* over rows to produce 3-D LSTM input.

    Returns
    -------
    X_seq : ndarray of shape ``(n - seq_len + 1, seq_len, features)``
    y_seq : ndarray of shape ``(n - seq_len + 1,)`` aligned to last row
    """
    n = len(X)
    if n < seq_len:
        raise ValueError(
            f"Input has {n} rows but seq_len={seq_len} — need at least {seq_len}"
        )
    X_seq = np.array([X[i : i + seq_len] for i in range(n - seq_len + 1)])
    y_seq = y[seq_len - 1 :]
    return X_seq, y_seq


class _LSTMNetwork(nn.Module):
    """Minimal LSTM for binary classification."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x)
        out = self.fc(h_n[-1])  # last layer hidden state
        return out.squeeze(-1)  # (batch,)


class LSTMClassifier:
    """sklearn-compatible wrapper for a PyTorch LSTM binary classifier."""

    def __init__(
        self,
        seq_len: int = 10,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
        lr: float = 1e-3,
        epochs: int = 50,
        batch_size: int = 32,
        random_state: int = 42,
    ) -> None:
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state

        self._scaler: StandardScaler | None = None
        self._network: _LSTMNetwork | None = None
        self._context_buffer: NDArray[np.floating] | None = None
        self._n_features: int = 0

    # ------------------------------------------------------------------
    # sklearn interface
    # ------------------------------------------------------------------

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LSTMClassifier":
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        self._n_features = X.shape[1]
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X.values)

        # Store context buffer for single-row prediction later
        self._context_buffer = X_scaled[-(self.seq_len - 1) :].copy()

        X_seq, y_seq = create_sequences(X_scaled, y.values, self.seq_len)
        X_t = torch.tensor(X_seq, dtype=torch.float32)
        y_t = torch.tensor(y_seq, dtype=torch.float32)

        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self._network = _LSTMNetwork(
            input_size=self._n_features,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
        )
        optimizer = torch.optim.Adam(self._network.parameters(), lr=self.lr)
        criterion = nn.BCEWithLogitsLoss()

        self._network.train()
        for _ in range(self.epochs):
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                logits = self._network(X_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()

        self._network.eval()
        return self

    def predict(self, X: pd.DataFrame) -> NDArray[np.integer]:
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> NDArray[np.floating]:
        if self._network is None or self._scaler is None:
            raise RuntimeError("Call fit() before predict_proba()")

        X_scaled = self._scaler.transform(X.values)

        # Prepend context buffer so we get exactly len(X) predictions
        if self._context_buffer is not None and len(X_scaled) < self.seq_len:
            needed = self.seq_len - len(X_scaled)
            prefix = self._context_buffer[-needed:]
            X_scaled = np.vstack([prefix, X_scaled])

        # If we still need context for a full-length X, prepend buffer
        if self._context_buffer is not None and len(X) >= self.seq_len:
            X_scaled = np.vstack([self._context_buffer, X_scaled])

        X_seq, _ = create_sequences(
            X_scaled,
            np.zeros(len(X_scaled), dtype=int),  # dummy y
            self.seq_len,
        )

        X_t = torch.tensor(X_seq, dtype=torch.float32)
        with torch.no_grad():
            logits = self._network(X_t)
            p = torch.sigmoid(logits).numpy()

        # Take only the last len(X) predictions (trim context prefix)
        p = p[-len(X) :]
        return np.column_stack([1.0 - p, p])

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {
            "seq_len": self.seq_len,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "lr": self.lr,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "random_state": self.random_state,
        }

    def set_params(self, **kwargs: Any) -> "LSTMClassifier":
        for key, value in kwargs.items():
            setattr(self, key, value)
        return self

    # ------------------------------------------------------------------
    # Pickling support (torch modules need special handling)
    # ------------------------------------------------------------------

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        if self._network is not None:
            state["_network_state_dict"] = self._network.state_dict()
            state["_network"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        network_state = state.pop("_network_state_dict", None)
        self.__dict__.update(state)
        if network_state is not None:
            self._network = _LSTMNetwork(
                input_size=self._n_features,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                dropout=self.dropout,
            )
            self._network.load_state_dict(network_state)
            self._network.eval()
