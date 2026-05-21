from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from xgboost import XGBClassifier


TARGET = "PitNextLap"
ID_COL = "id"


@dataclass(frozen=True)
class ModelConfig:
    name: str
    params: dict


MODEL_CONFIGS = [
    ModelConfig(
        name="xgb_depth6",
        params={
            "n_estimators": 420,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "min_child_weight": 3,
            "reg_lambda": 2.0,
            "gamma": 0.0,
        },
    ),
    ModelConfig(
        name="xgb_depth8",
        params={
            "n_estimators": 320,
            "max_depth": 8,
            "learning_rate": 0.045,
            "subsample": 0.9,
            "colsample_bytree": 0.8,
            "min_child_weight": 4,
            "reg_lambda": 3.0,
            "gamma": 0.15,
        },
    ),
]

TARGET_ENCODING_COLS = [
    "Driver",
    "Race",
    "RaceYear",
    "DriverCompound",
    "DriverRace",
]

CATEGORICAL_COLS = [
    "Driver",
    "Compound",
    "Race",
    "Year",
    "PitStop",
    "Stint",
    "RaceYear",
    "DriverCompound",
    "DriverRace",
]

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
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an F1 pit stop prediction model.")
    parser.add_argument("--train-path", type=Path, required=True)
    parser.add_argument("--test-path", type=Path, required=True)
    parser.add_argument(
        "--external-train-path",
        type=Path,
        default=None,
        help="Optional path to aligned real-world F1 training rows.",
    )
    parser.add_argument(
        "--external-weight",
        type=float,
        default=0.35,
        help="Sample weight for optional external training rows.",
    )
    parser.add_argument("--sample-submission-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--cv-mode",
        choices=["stratified", "group"],
        default="stratified",
        help="Validation splitter. `group` groups by RaceYear for a tougher check.",
    )
    return parser.parse_args()


def build_feature_frames(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    external_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    train = train_df.copy()
    test = test_df.copy()
    frames = [train.drop(columns=[TARGET]), test]
    frame_lengths = [len(train), len(test)]

    external = None
    if external_df is not None:
        external = external_df.copy()
        frames.append(external.drop(columns=[TARGET]))
        frame_lengths.append(len(external))

    combined = pd.concat(frames, axis=0, ignore_index=True, sort=False)

    combined["RaceYear"] = combined["Race"] + "|" + combined["Year"].astype(str)
    combined["DriverCompound"] = combined["Driver"] + "|" + combined["Compound"]
    combined["DriverRace"] = combined["Driver"] + "|" + combined["Race"]

    combined["EstimatedTotalLapsRaw"] = (
        combined["LapNumber"] / combined["RaceProgress"].replace(0, np.nan)
    )
    combined["EstimatedTotalLapsRaw"] = combined["EstimatedTotalLapsRaw"].clip(20, 90)

    race_year_totals = combined.groupby("RaceYear")["EstimatedTotalLapsRaw"].median()
    race_totals = combined.groupby("Race")["EstimatedTotalLapsRaw"].median()

    combined["EstimatedTotalLaps"] = combined["RaceYear"].map(race_year_totals)
    combined["EstimatedTotalLaps"] = combined["EstimatedTotalLaps"].fillna(
        combined["Race"].map(race_totals)
    )
    combined["EstimatedTotalLaps"] = combined["EstimatedTotalLaps"].fillna(
        combined["EstimatedTotalLapsRaw"]
    )
    combined["EstimatedTotalLaps"] = combined["EstimatedTotalLaps"].clip(20, 90)

    combined["LapsRemaining"] = (combined["EstimatedTotalLaps"] - combined["LapNumber"]).clip(lower=0)
    combined["TyreLifeRatio"] = combined["TyreLife"] / combined["EstimatedTotalLaps"].replace(0, np.nan)
    combined["TyreVsLapRatio"] = combined["TyreLife"] / combined["LapNumber"].replace(0, np.nan)
    combined["DegradationPerLap"] = (
        combined["Cumulative_Degradation"] / combined["TyreLife"].replace(0, np.nan)
    )
    combined["DeltaPerLap"] = combined["LapTime_Delta"] / combined["TyreLife"].replace(0, np.nan)
    combined["PositionPct"] = combined["Position"] / 20.0
    combined["StintTyreLoad"] = combined["Stint"] * combined["TyreLife"]
    combined["LateRaceTyreLoad"] = combined["TyreLife"] * combined["RaceProgress"]
    combined["TyreRemainingLoad"] = combined["TyreLife"] * combined["LapsRemaining"]
    combined["DeltaPositionInteraction"] = combined["LapTime_Delta"] * combined["PositionPct"]

    combined.replace([np.inf, -np.inf], np.nan, inplace=True)

    split_1 = frame_lengths[0]
    split_2 = split_1 + frame_lengths[1]

    train_features = combined.iloc[:split_1].copy()
    test_features = combined.iloc[split_1:split_2].copy()
    train_features[TARGET] = train[TARGET].values

    external_features = None
    if external is not None:
        external_features = combined.iloc[split_2:].copy()
        external_features[TARGET] = external[TARGET].values

    return train_features, test_features, external_features


def add_target_encoding(
    train_part: pd.DataFrame,
    valid_part: pd.DataFrame,
    test_part: pd.DataFrame,
    smoothing: float = 20.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_part = train_part.copy()
    valid_part = valid_part.copy()
    test_part = test_part.copy()
    global_mean = train_part[TARGET].mean()

    for col in TARGET_ENCODING_COLS:
        stats = train_part.groupby(col)[TARGET].agg(["sum", "count"])
        smooth = (stats["sum"] + smoothing * global_mean) / (stats["count"] + smoothing)
        feature_name = f"TE_{col}"
        train_part[feature_name] = train_part[col].map(smooth).fillna(global_mean)
        valid_part[feature_name] = valid_part[col].map(smooth).fillna(global_mean)
        test_part[feature_name] = test_part[col].map(smooth).fillna(global_mean)

    return train_part, valid_part, test_part


def finalize_matrices(
    train_part: pd.DataFrame,
    valid_part: pd.DataFrame,
    test_part: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    feature_cols = CATEGORICAL_COLS + NUMERIC_COLS + [f"TE_{col}" for col in TARGET_ENCODING_COLS]
    train_x = train_part[feature_cols].copy()
    valid_x = valid_part[feature_cols].copy()
    test_x = test_part[feature_cols].copy()

    for col in CATEGORICAL_COLS:
        train_vals = train_x[col].astype(str).fillna("__NA__")
        valid_vals = valid_x[col].astype(str).fillna("__NA__")
        test_vals = test_x[col].astype(str).fillna("__NA__")
        categories = pd.Index(pd.concat([train_vals, valid_vals, test_vals], axis=0).unique())
        train_x[col] = pd.Categorical(train_vals, categories=categories)
        valid_x[col] = pd.Categorical(valid_vals, categories=categories)
        test_x[col] = pd.Categorical(test_vals, categories=categories)

    for df in (train_x, valid_x, test_x):
        for col in NUMERIC_COLS + [f"TE_{c}" for c in TARGET_ENCODING_COLS]:
            df[col] = df[col].astype(float).fillna(-999.0)

    return train_x, valid_x, test_x


def build_model(config: ModelConfig, seed: int) -> XGBClassifier:
    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "enable_categorical": True,
        "random_state": seed,
        "n_jobs": 4,
        **config.params,
    }
    return XGBClassifier(**params)


def run_cv(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    external_df: pd.DataFrame | None,
    external_weight: float,
    folds: int,
    seed: int,
    cv_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    y = train_df[TARGET].astype(int)
    groups = train_df["RaceYear"]

    if cv_mode == "group":
        splitter = GroupKFold(n_splits=folds)
        split_iter = splitter.split(train_df, y, groups)
    else:
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        split_iter = splitter.split(train_df, y)

    oof = np.zeros(len(train_df), dtype=float)
    test_preds = np.zeros(len(test_df), dtype=float)
    total_models = len(MODEL_CONFIGS) * folds

    for model_idx, config in enumerate(MODEL_CONFIGS, start=1):
        fold_scores = []
        for fold_idx, (train_idx, valid_idx) in enumerate(split_iter, start=1):
            train_part = train_df.iloc[train_idx].copy()
            valid_part = train_df.iloc[valid_idx].copy()
            test_part = test_df.copy()
            train_part["_sample_weight"] = 1.0

            if external_df is not None:
                external_part = external_df.copy()
                external_part["_sample_weight"] = external_weight
                train_part = pd.concat([train_part, external_part], axis=0, ignore_index=True)

            train_part, valid_part, test_part = add_target_encoding(train_part, valid_part, test_part)
            train_x, valid_x, test_x = finalize_matrices(train_part, valid_part, test_part)

            model = build_model(config, seed + model_idx + fold_idx)
            model.fit(
                train_x,
                train_part[TARGET].astype(int),
                sample_weight=train_part["_sample_weight"].astype(float).values,
                verbose=False,
            )

            valid_pred = model.predict_proba(valid_x)[:, 1]
            test_pred = model.predict_proba(test_x)[:, 1]

            oof[valid_idx] += valid_pred / len(MODEL_CONFIGS)
            test_preds += test_pred / total_models

            fold_auc = roc_auc_score(valid_part[TARGET].astype(int), valid_pred)
            fold_scores.append(fold_auc)
            print(
                f"[{config.name}] fold {fold_idx}/{folds} auc={fold_auc:.6f}"
            )

        print(f"[{config.name}] mean auc={np.mean(fold_scores):.6f}")

        if cv_mode == "group":
            split_iter = GroupKFold(n_splits=folds).split(train_df, y, groups)
        else:
            split_iter = StratifiedKFold(
                n_splits=folds, shuffle=True, random_state=seed
            ).split(train_df, y)

    full_auc = roc_auc_score(y, oof)
    print(f"Overall OOF AUC: {full_auc:.6f}")
    return oof, test_preds


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

    train_df, test_df, external_df = build_feature_frames(train_raw, test_raw, external_raw)

    oof_pred, test_pred = run_cv(
        train_df=train_df,
        test_df=test_df,
        external_df=external_df,
        external_weight=args.external_weight,
        folds=args.folds,
        seed=args.seed,
        cv_mode=args.cv_mode,
    )

    oof_path = args.output_dir / "oof_predictions.csv"
    pd.DataFrame(
        {
            ID_COL: train_raw[ID_COL],
            TARGET: train_raw[TARGET],
            "prediction": oof_pred,
        }
    ).to_csv(oof_path, index=False)

    if args.sample_submission_path and args.sample_submission_path.exists():
        submission = pd.read_csv(args.sample_submission_path)
        submission[TARGET] = test_pred
    else:
        submission = pd.DataFrame({ID_COL: test_raw[ID_COL], TARGET: test_pred})

    submission_path = args.output_dir / "submission.csv"
    submission.to_csv(submission_path, index=False)

    print(f"Saved OOF predictions to {oof_path}")
    print(f"Saved submission to {submission_path}")


if __name__ == "__main__":
    main()
