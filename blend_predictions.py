from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Blend OOF and submission predictions.")
    parser.add_argument("--oof-a", type=Path, required=True)
    parser.add_argument("--oof-b", type=Path, required=True)
    parser.add_argument("--sub-a", type=Path, required=True)
    parser.add_argument("--sub-b", type=Path, required=True)
    parser.add_argument("--weight-a", type=float, default=0.5)
    parser.add_argument("--weight-b", type=float, default=0.5)
    parser.add_argument("--rank-average", action="store_true")
    parser.add_argument("--output-path", type=Path, required=True)
    return parser.parse_args()


def blend_series(a: pd.Series, b: pd.Series, weight_a: float, weight_b: float, rank_average: bool) -> pd.Series:
    if rank_average:
        a = a.rank(pct=True)
        b = b.rank(pct=True)
    denom = weight_a + weight_b
    return (weight_a * a + weight_b * b) / denom


def main() -> None:
    args = parse_args()
    oof_a = pd.read_csv(args.oof_a)
    oof_b = pd.read_csv(args.oof_b)
    sub_a = pd.read_csv(args.sub_a)
    sub_b = pd.read_csv(args.sub_b)

    merged_oof = oof_a.merge(oof_b[["id", "prediction"]], on="id", suffixes=("_a", "_b"))
    merged_sub = sub_a.merge(sub_b, on="id", suffixes=("_a", "_b"))

    blended_oof = blend_series(
        merged_oof["prediction_a"],
        merged_oof["prediction_b"],
        args.weight_a,
        args.weight_b,
        args.rank_average,
    )
    auc = roc_auc_score(merged_oof["PitNextLap"], blended_oof)
    print(f"Blended OOF AUC: {auc:.6f}")

    output = pd.DataFrame(
        {
            "id": merged_sub["id"],
            "PitNextLap": blend_series(
                merged_sub["PitNextLap_a"],
                merged_sub["PitNextLap_b"],
                args.weight_a,
                args.weight_b,
                args.rank_average,
            ),
        }
    )
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_path, index=False)
    print(f"Saved blended submission to {args.output_path}")


if __name__ == "__main__":
    main()
