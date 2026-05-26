from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ID_COL = "id"
TARGET_COL = "PitNextLap"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a tiny weighted blend between two submission files."
    )
    parser.add_argument("--primary", type=Path, required=True, help="Dominant submission file.")
    parser.add_argument("--secondary", type=Path, required=True, help="Secondary submission file.")
    parser.add_argument("--primary-weight", type=float, default=0.99)
    parser.add_argument("--secondary-weight", type=float, default=0.01)
    parser.add_argument("--rank-space", action="store_true", help="Blend percentile ranks instead of raw probabilities.")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_submission(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if ID_COL not in df.columns:
        raise ValueError(f"{path} is missing '{ID_COL}'")
    value_cols = [col for col in df.columns if col != ID_COL]
    if not value_cols:
        raise ValueError(f"{path} has no prediction column")
    pred_col = TARGET_COL if TARGET_COL in df.columns else value_cols[0]
    df = df[[ID_COL, pred_col]].rename(columns={pred_col: TARGET_COL})
    if df[ID_COL].duplicated().any():
        raise ValueError(f"{path} contains duplicate ids")
    values = pd.to_numeric(df[TARGET_COL], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{path} contains non-finite prediction values")
    df[TARGET_COL] = np.clip(values, 0.0, 1.0)
    return df


def main() -> None:
    args = parse_args()
    primary = read_submission(args.primary)
    secondary = read_submission(args.secondary)

    merged = primary.merge(
        secondary,
        on=ID_COL,
        suffixes=("_primary", "_secondary"),
        validate="one_to_one",
    )

    p = merged[f"{TARGET_COL}_primary"]
    s = merged[f"{TARGET_COL}_secondary"]

    if args.rank_space:
        p = p.rank(pct=True)
        s = s.rank(pct=True)

    denom = args.primary_weight + args.secondary_weight
    merged[TARGET_COL] = np.clip(
        (args.primary_weight * p + args.secondary_weight * s) / denom,
        0.0,
        1.0,
    )

    output = merged[[ID_COL, TARGET_COL]]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)

    print(f"Saved {args.output}")
    print(f"rows: {len(output)}")
    print(f"mean: {output[TARGET_COL].mean():.8f}")
    print(f"std: {output[TARGET_COL].std():.8f}")
    print(f"min: {output[TARGET_COL].min():.8f}")
    print(f"max: {output[TARGET_COL].max():.8f}")


if __name__ == "__main__":
    main()
