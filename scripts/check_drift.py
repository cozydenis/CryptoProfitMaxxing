"""Check for data drift in the feature pipeline.

Usage:
    python scripts/check_drift.py
    python scripts/check_drift.py --test-window 60 --p-val 0.01

Exit codes: 0 = no drift, 1 = drift detected.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.drift.detector import check_drift_from_features


def _load_params() -> dict:
    params_path = Path(__file__).resolve().parent.parent / "params.yaml"
    with params_path.open() as f:
        return yaml.safe_load(f).get("drift", {})


def main() -> int:
    defaults = _load_params()
    parser = argparse.ArgumentParser(description="Check for data drift")
    parser.add_argument(
        "--features",
        default=defaults.get("features_path", "data/processed/features.csv"),
    )
    parser.add_argument(
        "--test-frac", type=float, default=defaults.get("test_frac", 0.2),
    )
    parser.add_argument(
        "--test-window", type=int, default=defaults.get("test_window", 30),
    )
    parser.add_argument(
        "--p-val", type=float, default=defaults.get("p_val", 0.05),
    )
    args = parser.parse_args()

    result = check_drift_from_features(
        args.features,
        test_frac=args.test_frac,
        test_window=args.test_window,
        p_val=args.p_val,
    )

    print(f"\n{'Feature':<20} {'p-value':>10} {'Drift?':>8}")
    print("-" * 40)
    for name, pv, drifted in zip(
        result.feature_names, result.p_values, result.is_drift_per_feature
    ):
        flag = "YES" if drifted else "no"
        print(f"{name:<20} {pv:>10.4f} {flag:>8}")

    print("-" * 40)
    print(
        f"Reference: {result.n_reference} rows | "
        f"Test window: {result.n_test} rows | "
        f"Threshold: {result.p_val_threshold}"
    )

    if result.is_drift:
        print("\n** DRIFT DETECTED **\n")
        return 1

    print("\nNo drift detected.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
