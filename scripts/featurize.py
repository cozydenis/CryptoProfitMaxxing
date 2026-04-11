"""CLI entrypoint: read raw CSV, add indicators + target, write features CSV."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features.indicators import add_indicators
from src.features.labels import add_target


def _load_params() -> dict:
    params_path = Path(__file__).resolve().parent.parent / "params.yaml"
    with params_path.open() as f:
        return yaml.safe_load(f)["featurize"]


def main() -> int:
    defaults = _load_params()
    parser = argparse.ArgumentParser(description="Compute features and targets")
    parser.add_argument("--in", dest="in_path", default=defaults["in_path"])
    parser.add_argument("--out", default=defaults["out_path"])
    parser.add_argument("--horizon", type=int, default=defaults["horizon"])
    args = parser.parse_args()

    raw = pd.read_csv(args.in_path, parse_dates=["timestamp"])
    with_indicators = add_indicators(raw)
    with_target = add_target(with_indicators, horizon=args.horizon)
    clean = with_target.dropna().reset_index(drop=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(out_path, index=False)
    print(f"Wrote {len(clean)} rows to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
