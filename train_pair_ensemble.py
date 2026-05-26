from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier


TARGET = "PitNextLap"
ID_COL = "id"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an XGBoost + LightGBM pair ensemble for F1 pit-stop prediction.")
    parser.add_argument("--train-path", type=Path, required=True)
    parser.add_argument("--test-path", type=Path, required=True)
    parser.add_argument("--sample-submission-path", type=Path, default=None)
    parser.add_argument("--external-train-path", type=Path, default=None)
    parser.add_argument("--external-weight", type=float, default=0.35)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs_pair"))
    return parser.parse_args()


CAT_COLS = [
    "Driver",
    "Compound",
    "Race",
    "Year",
    "PitStop",
    "Stint",
    "RaceYear",
    "DriverRace",
    "DriverCompound",
    "CompoundStint",
    "CompoundRace",
    "CompoundYear",
    "RaceStint",
    "CompoundRaceYear",
]

TE_COLS = [
    "Driver",
    "Race",
    "RaceYear",
    "DriverCompound",
    "CompoundStint",
    "CompoundRaceYear",
    "TyreLifeBin4",
    "LikelyLastStint",
    "PittedLastObservedLap",
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["_orig_order"] = np.arange(len(df))
    df["RaceYear"] = df["Race"].astype(str) + "|" + df["Year"].astype(str)
    df["DriverRace"] = df["Driver"].astype(str) + "|" + df["Race"].astype(str)
    df["DriverCompound"] = df["Driver"].astype(str) + "|" + df["Compound"].astype(str)
    df["CompoundStint"] = df["Compound"].astype(str) + "|" + df["Stint"].astype(str)

    estimated_total = (df["LapNumber"] / df["RaceProgress"].clip(lower=1e-3)).clip(20, 90)
    df["EstimatedTotalLaps"] = estimated_total
    df["LapsRemaining"] = (estimated_total - df["LapNumber"]).clip(lower=0)

    df["TyreAgePerLap"] = df["TyreLife"] / df["LapNumber"].clip(lower=1)
    df["DegRate"] = df["Cumulative_Degradation"] / df["TyreLife"].clip(lower=1)
    df["PittedLastObservedLap"] = (df["TyreLife"] <= 1).astype("int8")
    df["TyreLifeBin4"] = (
        pd.cut(df["TyreLife"], bins=[0, 20, 40, 60, 80], labels=False, include_lowest=True)
        .astype("float")
        .fillna(0)
        .astype(int)
    )
    df["LapTimePerTyre"] = df["LapTime (s)"] / df["TyreLife"].clip(lower=1)
    df["RaceProgressRemaining"] = 1.0 - df["RaceProgress"]
    df["LapTimeXDeg"] = df["LapTime (s)"] * df["Cumulative_Degradation"]
    df["LapDeltaAbs"] = df["LapTime_Delta"].abs()

    group_cols = ["Driver", "Race", "Year", "Stint"]
    df = df.sort_values(group_cols + ["LapNumber"]).reset_index(drop=True)
    df["LapInStintObs"] = df.groupby(group_cols).cumcount() + 1
    df["LapsSinceLastPitObs"] = df.groupby(group_cols).cumcount()
    df["LikelyLastStint"] = (df["RaceProgressRemaining"] < 0.15).astype("int8")

    df["CompoundRace"] = df["Compound"].astype(str) + "|" + df["Race"].astype(str)
    df["CompoundYear"] = df["Compound"].astype(str) + "|" + df["Year"].astype(str)
    df["RaceStint"] = df["Race"].astype(str) + "|" + df["Stint"].astype(str)
    df["CompoundRaceYear"] = (
        df["Compound"].astype(str) + "|" + df["Race"].astype(str) + "|" + df["Year"].astype(str)
    )
    df = df.sort_values("_orig_order").drop(columns=["_orig_order"]).reset_index(drop=True)
    return df.replace([np.inf, -np.inf], np.nan)


def build_category_maps(train_df: pd.DataFrame, test_df: pd.DataFrame, external_df: pd.DataFrame) -> dict[str, dict[str, int]]:
    all_cats = pd.concat([train_df[CAT_COLS], test_df[CAT_COLS], external_df[CAT_COLS]], axis=0, ignore_index=True)
    return {
        col: {value: idx for idx, value in enumerate(all_cats[col].astype(str).fillna("__NA__").unique())}
        for col in CAT_COLS
    }


def add_target_encodings(
    train_ref: pd.DataFrame,
    train_aug: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_aug = train_aug.copy()
    valid_df = valid_df.copy()
    test_df = test_df.copy()
    global_mean = train_ref[TARGET].mean()

    for col in TE_COLS:
        stats = train_ref.groupby(col)[TARGET].agg(["sum", "count"])
        smooth = (stats["sum"] + 20 * global_mean) / (stats["count"] + 20)
        for df in (train_aug, valid_df, test_df):
            df[f"TE_{col}"] = df[col].map(smooth).fillna(global_mean)

    return train_aug, valid_df, test_df


def encode_frame(df: pd.DataFrame, category_maps: dict[str, dict[str, int]], feature_cols: list[str], num_cols: list[str]) -> pd.DataFrame:
    X = df[feature_cols].copy()
    for col in CAT_COLS:
        X[col] = df[col].astype(str).fillna("__NA__").map(category_maps[col]).astype("int32")
    for col in num_cols:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(-999.0).astype("float32")
    return X


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_raw = pd.read_csv(args.train_path)
    test_raw = pd.read_csv(args.test_path)
    external_raw = pd.read_csv(args.external_train_path) if args.external_train_path else None
    if external_raw is None:
        raise ValueError("--external-train-path is required for this pipeline.")
    drop_cols = [col for col in external_raw.columns if col not in train_raw.columns]
    if drop_cols:
        external_raw = external_raw.drop(columns=drop_cols)

    train_fe = engineer_features(train_raw)
    test_fe = engineer_features(test_raw)
    external_fe = engineer_features(external_raw)

    common_features = sorted(
        set(train_fe.columns) & set(test_fe.columns) & set(external_fe.columns) - {TARGET}
    )
    train_df = train_fe[common_features + [TARGET]].copy()
    test_df = test_fe[common_features].copy()
    external_df = external_fe[common_features + [TARGET]].copy()

    num_cols = [col for col in common_features if col not in CAT_COLS + [ID_COL]]
    te_feature_cols = [f"TE_{col}" for col in TE_COLS]
    feature_cols = CAT_COLS + num_cols + te_feature_cols

    y = train_df[TARGET].astype(int)
    category_maps = build_category_maps(train_df, test_df, external_df)

    splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    xgb_oof = np.zeros(len(train_df), dtype=float)
    lgb_oof = np.zeros(len(train_df), dtype=float)
    xgb_test = np.zeros(len(test_df), dtype=float)
    lgb_test = np.zeros(len(test_df), dtype=float)

    for fold_idx, (train_idx, valid_idx) in enumerate(splitter.split(train_df, y), start=1):
        base_train = train_df.iloc[train_idx].copy()
        valid_df = train_df.iloc[valid_idx].copy()
        test_part = test_df.copy()
        external_part = external_df.copy()

        base_train["_sample_weight"] = 1.0
        external_part["_sample_weight"] = args.external_weight
        train_aug = pd.concat([base_train, external_part], axis=0, ignore_index=True)

        train_aug, valid_df, test_part = add_target_encodings(base_train, train_aug, valid_df, test_part)
        X_train = encode_frame(train_aug, category_maps, feature_cols, num_cols + te_feature_cols)
        X_valid = encode_frame(valid_df, category_maps, feature_cols, num_cols + te_feature_cols)
        X_test = encode_frame(test_part, category_maps, feature_cols, num_cols + te_feature_cols)

        y_train = train_aug[TARGET].astype(int)
        y_valid = valid_df[TARGET].astype(int)
        sample_weight = train_aug["_sample_weight"].astype(float).values

        xgb_model = XGBClassifier(
            n_estimators=400,
            learning_rate=0.03,
            max_depth=9,
            subsample=0.9,
            colsample_bytree=0.70,
            min_child_weight=18,
            reg_alpha=3.0,
            reg_lambda=2.7,
            gamma=0.3,
            objective="binary:logistic",
            eval_metric="auc",
            tree_method="hist",
            random_state=args.seed + fold_idx,
            n_jobs=4,
            verbosity=0,
        )
        lgb_model = LGBMClassifier(
            n_estimators=900,
            learning_rate=0.04,
            max_depth=10,
            num_leaves=113,
            min_child_samples=28,
            subsample=0.60,
            colsample_bytree=0.55,
            reg_alpha=3.8,
            reg_lambda=0.12,
            objective="binary",
            random_state=args.seed + fold_idx,
            n_jobs=4,
            verbose=-1,
        )

        xgb_model.fit(X_train, y_train, sample_weight=sample_weight)
        lgb_model.fit(X_train, y_train, sample_weight=sample_weight)

        xgb_valid = xgb_model.predict_proba(X_valid)[:, 1]
        lgb_valid = lgb_model.predict_proba(X_valid)[:, 1]
        xgb_test += xgb_model.predict_proba(X_test)[:, 1] / args.folds
        lgb_test += lgb_model.predict_proba(X_test)[:, 1] / args.folds

        xgb_oof[valid_idx] = xgb_valid
        lgb_oof[valid_idx] = lgb_valid

        print(
            f"fold {fold_idx}/{args.folds} "
            f"xgb_auc={roc_auc_score(y_valid, xgb_valid):.6f} "
            f"lgb_auc={roc_auc_score(y_valid, lgb_valid):.6f} "
            f"avg_auc={roc_auc_score(y_valid, 0.5 * xgb_valid + 0.5 * lgb_valid):.6f}"
        )

    meta_train = pd.DataFrame({"xgb": xgb_oof, "lgb": lgb_oof})
    meta_test = pd.DataFrame({"xgb": xgb_test, "lgb": lgb_test})
    meta_model = LogisticRegression(max_iter=1000)
    meta_model.fit(meta_train, y)
    stack_oof = meta_model.predict_proba(meta_train)[:, 1]
    stack_test = meta_model.predict_proba(meta_test)[:, 1]

    print(f"overall_xgb_auc={roc_auc_score(y, xgb_oof):.6f}")
    print(f"overall_lgb_auc={roc_auc_score(y, lgb_oof):.6f}")
    print(f"overall_avg_auc={roc_auc_score(y, 0.5 * xgb_oof + 0.5 * lgb_oof):.6f}")
    print(f"overall_pair_stack_auc={roc_auc_score(y, stack_oof):.6f}")
    print(f"meta_coef={meta_model.coef_.tolist()} intercept={meta_model.intercept_.tolist()}")

    pd.DataFrame(
        {
            ID_COL: train_raw[ID_COL],
            TARGET: train_raw[TARGET],
            "xgb": xgb_oof,
            "lgb": lgb_oof,
            "stack_prediction": stack_oof,
        }
    ).to_csv(args.output_dir / "oof_predictions.csv", index=False)

    if args.sample_submission_path and args.sample_submission_path.exists():
        submission = pd.read_csv(args.sample_submission_path)
        submission[TARGET] = stack_test
    else:
        submission = pd.DataFrame({ID_COL: test_raw[ID_COL], TARGET: stack_test})

    submission.to_csv(args.output_dir / "submission.csv", index=False)


if __name__ == "__main__":
    main()
