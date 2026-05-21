from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


TARGET = "PitNextLap"
ID_COL = "id"

HIGH_CARD_COLS = [
    "Driver",
    "Race",
    "RaceYear",
    "DriverCompound",
    "DriverRace",
    "CompoundStint",
]

SMALL_ONEHOT_COLS = ["Compound", "Year", "PitStop", "Stint"]

NUMERIC_COLS = [
    "LapNumber",
    "TyreLife",
    "Position",
    "LapTime (s)",
    "LapTime_Delta",
    "Cumulative_Degradation",
    "RaceProgress",
    "Position_Change",
    "EstimatedTotalLaps",
    "LapsRemaining",
    "TyreLifeRatio",
    "TyreVsLapRatio",
    "DegradationPerLap",
    "DeltaPerLap",
    "PositionPct",
    "StintTyreLoad",
    "LateRaceTyreLoad",
    "TyreRemainingLoad",
    "DeltaPositionInteraction",
    "LapTyreInteraction",
    "ProgressSquared",
    "TyreLifeSquared",
]

VOTE_BASE_MODELS = ["catboost", "xgboost", "histgb", "lightgbm"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train stacked models for F1 pit stop prediction.")
    parser.add_argument("--train-path", type=Path, required=True)
    parser.add_argument("--test-path", type=Path, required=True)
    parser.add_argument("--sample-submission-path", type=Path, default=None)
    parser.add_argument("--external-train-path", type=Path, default=None)
    parser.add_argument("--external-frac", type=float, default=0.35)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--base-models",
        type=str,
        default="catboost,xgboost,histgb,lightgbm,knn,logreg,et",
        help="Comma-separated base model names.",
    )
    parser.add_argument(
        "--meta-model",
        type=str,
        default="logreg",
        choices=["logreg", "histgb"],
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs_stacking"))
    return parser.parse_args()


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["RaceYear"] = df["Race"] + "|" + df["Year"].astype(str)
    df["DriverCompound"] = df["Driver"] + "|" + df["Compound"]
    df["DriverRace"] = df["Driver"] + "|" + df["Race"]
    df["CompoundStint"] = df["Compound"] + "|" + df["Stint"].astype(str)

    df["EstimatedTotalLapsRaw"] = (
        df["LapNumber"] / df["RaceProgress"].replace(0, np.nan)
    ).clip(20, 90)
    race_year_totals = df.groupby("RaceYear")["EstimatedTotalLapsRaw"].transform("median")
    df["EstimatedTotalLaps"] = race_year_totals.fillna(df["EstimatedTotalLapsRaw"])
    df["LapsRemaining"] = (df["EstimatedTotalLaps"] - df["LapNumber"]).clip(lower=0)

    df["TyreLifeRatio"] = df["TyreLife"] / df["EstimatedTotalLaps"].replace(0, np.nan)
    df["TyreVsLapRatio"] = df["TyreLife"] / df["LapNumber"].replace(0, np.nan)
    df["DegradationPerLap"] = df["Cumulative_Degradation"] / df["TyreLife"].replace(0, np.nan)
    df["DeltaPerLap"] = df["LapTime_Delta"] / df["TyreLife"].replace(0, np.nan)
    df["PositionPct"] = df["Position"] / 20.0
    df["StintTyreLoad"] = df["Stint"] * df["TyreLife"]
    df["LateRaceTyreLoad"] = df["TyreLife"] * df["RaceProgress"]
    df["TyreRemainingLoad"] = df["TyreLife"] * df["LapsRemaining"]
    df["DeltaPositionInteraction"] = df["LapTime_Delta"] * df["PositionPct"]
    df["LapTyreInteraction"] = df["LapNumber"] * df["TyreLife"]
    df["ProgressSquared"] = df["RaceProgress"] ** 2
    df["TyreLifeSquared"] = df["TyreLife"] ** 2
    return df.replace([np.inf, -np.inf], np.nan)


def add_fold_encodings(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = train_df.copy()
    valid_df = valid_df.copy()
    test_df = test_df.copy()
    global_mean = train_df[TARGET].mean()

    for col in HIGH_CARD_COLS:
        stats = train_df.groupby(col)[TARGET].agg(["sum", "count"])
        smooth = (stats["sum"] + 20 * global_mean) / (stats["count"] + 20)
        freq = train_df[col].value_counts(normalize=True)
        for df in (train_df, valid_df, test_df):
            df[f"TE_{col}"] = df[col].map(smooth).fillna(global_mean)
            df[f"FE_{col}"] = df[col].map(freq).fillna(0.0)
    return train_df, valid_df, test_df


def to_dense_matrix(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dense_cols = (
        NUMERIC_COLS
        + [f"TE_{col}" for col in HIGH_CARD_COLS]
        + [f"FE_{col}" for col in HIGH_CARD_COLS]
    )

    def onehot(df: pd.DataFrame) -> pd.DataFrame:
        return pd.get_dummies(df[SMALL_ONEHOT_COLS].astype(str), prefix=SMALL_ONEHOT_COLS, dtype=float)

    train_small = onehot(train_df)
    valid_small = onehot(valid_df)
    test_small = onehot(test_df)
    train_small, valid_small = train_small.align(valid_small, join="outer", axis=1, fill_value=0.0)
    train_small, test_small = train_small.align(test_small, join="outer", axis=1, fill_value=0.0)
    valid_small, test_small = valid_small.align(test_small, join="outer", axis=1, fill_value=0.0)

    train_x = pd.concat([train_df[dense_cols].reset_index(drop=True), train_small.reset_index(drop=True)], axis=1)
    valid_x = pd.concat([valid_df[dense_cols].reset_index(drop=True), valid_small.reset_index(drop=True)], axis=1)
    test_x = pd.concat([test_df[dense_cols].reset_index(drop=True), test_small.reset_index(drop=True)], axis=1)
    return train_x, valid_x, test_x


def build_base_model(name: str, seed: int):
    if name == "catboost":
        return CatBoostClassifier(
            iterations=500,
            depth=8,
            learning_rate=0.05,
            loss_function="Logloss",
            eval_metric="AUC",
            verbose=False,
            random_seed=seed,
        )
    if name == "xgboost":
        return XGBClassifier(
            n_estimators=350,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.82,
            min_child_weight=4,
            reg_lambda=3.0,
            gamma=0.1,
            objective="binary:logistic",
            eval_metric="auc",
            tree_method="hist",
            random_state=seed,
            n_jobs=4,
        )
    if name == "lightgbm":
        return LGBMClassifier(
            n_estimators=450,
            learning_rate=0.04,
            num_leaves=63,
            max_depth=-1,
            min_child_samples=30,
            subsample=0.9,
            colsample_bytree=0.85,
            reg_lambda=2.0,
            random_state=seed,
            n_jobs=4,
            objective="binary",
        )
    if name == "histgb":
        return HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_depth=8,
            max_leaf_nodes=63,
            min_samples_leaf=30,
            l2_regularization=0.1,
            random_state=seed,
        )
    if name == "knn":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="constant", fill_value=-999.0)),
                ("scaler", StandardScaler()),
                ("knn", KNeighborsClassifier(n_neighbors=120, weights="distance", metric="minkowski", p=2)),
            ]
        )
    if name == "logreg":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="constant", fill_value=-999.0)),
                ("scaler", StandardScaler()),
                ("logreg", LogisticRegression(max_iter=600, C=0.6)),
            ]
        )
    if name == "et":
        return ExtraTreesClassifier(
            n_estimators=500,
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=4,
        )
    raise ValueError(f"Unknown model name: {name}")


def build_meta_model(name: str, seed: int):
    if name == "logreg":
        return LogisticRegression(max_iter=800, C=1.0)
    return HistGradientBoostingClassifier(
        learning_rate=0.03,
        max_depth=3,
        max_leaf_nodes=15,
        min_samples_leaf=50,
        random_state=seed,
    )


def predict_prob(model, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def fit_model(name: str, model, X: pd.DataFrame, y: pd.Series, sample_weight: np.ndarray | None) -> None:
    if name in {"catboost", "xgboost", "lightgbm", "histgb"} and sample_weight is not None:
        model.fit(X, y, sample_weight=sample_weight)
        return
    model.fit(X, y)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_raw = pd.read_csv(args.train_path)
    test_raw = pd.read_csv(args.test_path)
    external_raw = None
    if args.external_train_path and args.external_train_path.exists():
        external_raw = pd.read_csv(args.external_train_path)
        drop_cols = [col for col in external_raw.columns if col not in train_raw.columns]
        if drop_cols:
            external_raw = external_raw.drop(columns=drop_cols)

    train_df = engineer_features(train_raw)
    test_df = engineer_features(test_raw)
    external_df = engineer_features(external_raw) if external_raw is not None else None

    base_models = [name.strip() for name in args.base_models.split(",") if name.strip()]
    y = train_df[TARGET].astype(int)
    splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)

    oof_base = {name: np.zeros(len(train_df), dtype=float) for name in base_models}
    test_base = {name: np.zeros(len(test_df), dtype=float) for name in base_models}

    for fold_idx, (train_idx, valid_idx) in enumerate(splitter.split(train_df, y), start=1):
        fold_train = train_df.iloc[train_idx].copy()
        fold_valid = train_df.iloc[valid_idx].copy()
        fold_test = test_df.copy()
        fold_train["_sample_weight"] = 1.0

        if external_df is not None and args.external_frac > 0:
            ext_weighted = external_df.copy()
            ext_weighted["_sample_weight"] = args.external_frac
            fold_train = pd.concat([fold_train, ext_weighted], axis=0, ignore_index=True)

        fold_train, fold_valid, fold_test = add_fold_encodings(fold_train, fold_valid, fold_test)
        train_x, valid_x, test_x = to_dense_matrix(fold_train, fold_valid, fold_test)
        train_y = fold_train[TARGET].astype(int)
        valid_y = fold_valid[TARGET].astype(int)
        train_w = fold_train["_sample_weight"].astype(float).values

        for model_name in base_models:
            model = build_base_model(model_name, args.seed + fold_idx)
            fit_model(model_name, model, train_x, train_y, train_w)
            valid_pred = predict_prob(model, valid_x)
            test_pred = predict_prob(model, test_x)
            oof_base[model_name][valid_idx] = valid_pred
            test_base[model_name] += test_pred / args.folds
            fold_auc = roc_auc_score(valid_y, valid_pred)
            print(f"[{model_name}] fold {fold_idx}/{args.folds} auc={fold_auc:.6f}")

    oof_frame = pd.DataFrame({ID_COL: train_raw[ID_COL], TARGET: y})
    test_frame = pd.DataFrame({ID_COL: test_raw[ID_COL]})
    for model_name in base_models:
        oof_frame[model_name] = oof_base[model_name]
        test_frame[model_name] = test_base[model_name]
        model_auc = roc_auc_score(y, oof_base[model_name])
        print(f"[{model_name}] overall oof auc={model_auc:.6f}")

    vote_members = [name for name in VOTE_BASE_MODELS if name in base_models]
    if vote_members:
        oof_frame["vote4"] = oof_frame[vote_members].mean(axis=1)
        test_frame["vote4"] = test_frame[vote_members].mean(axis=1)
        vote_auc = roc_auc_score(y, oof_frame["vote4"])
        print(f"[vote4] overall oof auc={vote_auc:.6f}")

    meta_features = [name for name in base_models if name != args.meta_model]
    if "vote4" in oof_frame.columns:
        meta_features.append("vote4")

    meta_model = build_meta_model(args.meta_model, args.seed)
    meta_model.fit(oof_frame[meta_features], y)
    stack_oof = predict_prob(meta_model, oof_frame[meta_features])
    stack_test = predict_prob(meta_model, test_frame[meta_features])
    stack_auc = roc_auc_score(y, stack_oof)
    print(f"[stack_{args.meta_model}] overall oof auc={stack_auc:.6f}")

    oof_frame["stack_prediction"] = stack_oof
    oof_frame.to_csv(args.output_dir / "oof_base_predictions.csv", index=False)

    if args.sample_submission_path and args.sample_submission_path.exists():
        submission = pd.read_csv(args.sample_submission_path)
        submission[TARGET] = stack_test
    else:
        submission = pd.DataFrame({ID_COL: test_raw[ID_COL], TARGET: stack_test})
    submission.to_csv(args.output_dir / "submission_stack.csv", index=False)

    for model_name in base_models:
        per_model_sub = pd.DataFrame({ID_COL: test_raw[ID_COL], TARGET: test_base[model_name]})
        per_model_sub.to_csv(args.output_dir / f"submission_{model_name}.csv", index=False)


if __name__ == "__main__":
    main()
