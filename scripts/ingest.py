"""CLI entrypoint: fetch CoinGecko market chart and write CSV."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion.coingecko import fetch_market_chart, validate_frame


def _load_params() -> dict:
    params_path = Path(__file__).resolve().parent.parent / "params.yaml"
    with params_path.open() as f:
        return yaml.safe_load(f)["ingest"]


def main() -> int:
    defaults = _load_params()
    parser = argparse.ArgumentParser(description="Ingest CoinGecko market chart")
    parser.add_argument("--coin", default=defaults["coin_id"])
    parser.add_argument("--vs", default=defaults["vs_currency"])
    parser.add_argument("--days", type=int, default=defaults["days"])
    parser.add_argument("--out", default=defaults["out_path"])
    args = parser.parse_args()

    df = fetch_market_chart(coin_id=args.coin, vs_currency=args.vs, days=args.days)
    validate_frame(df)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
