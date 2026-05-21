from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train dense tabular models for F1 pit stop prediction.")
    parser.add_argument("--train-path", type=Path, required=True)
    parser.add_argument("--test-path", type=Path, required=True)
    parser.add_argument("--sample-submission-path", type=Path, default=None)
    parser.add_argument("--external-train-path", type=Path, default=None)
    parser.add_argument("--external-frac", type=float, default=0.35)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--models", type=str, default="et", help="Comma-separated: et,mlp,logreg")
    parser.add_argument("--corr-threshold", type=float, default=1.1, help="Set below 1.0 to drop highly correlated dense features.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs_dense"))
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

    def build_onehots(df: pd.DataFrame) -> pd.DataFrame:
        return pd.get_dummies(df[SMALL_ONEHOT_COLS].astype(str), prefix=SMALL_ONEHOT_COLS, dtype=float)

    train_small = build_onehots(train_df)
    valid_small = build_onehots(valid_df)
    test_small = build_onehots(test_df)

    train_small, valid_small = train_small.align(valid_small, join="outer", axis=1, fill_value=0.0)
    train_small, test_small = train_small.align(test_small, join="outer", axis=1, fill_value=0.0)
    valid_small, test_small = valid_small.align(test_small, join="outer", axis=1, fill_value=0.0)

    train_x = pd.concat([train_df[dense_cols].reset_index(drop=True), train_small.reset_index(drop=True)], axis=1)
    valid_x = pd.concat([valid_df[dense_cols].reset_index(drop=True), valid_small.reset_index(drop=True)], axis=1)
    test_x = pd.concat([test_df[dense_cols].reset_index(drop=True), test_small.reset_index(drop=True)], axis=1)
    return train_x, valid_x, test_x


def correlation_filter(
    train_x: pd.DataFrame,
    valid_x: pd.DataFrame,
    test_x: pd.DataFrame,
    threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if threshold >= 1.0:
        return train_x, valid_x, test_x

    corr = train_x.corr(numeric_only=True).abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop_cols = [col for col in upper.columns if any(upper[col] > threshold)]
    return (
        train_x.drop(columns=drop_cols),
        valid_x.drop(columns=drop_cols),
        test_x.drop(columns=drop_cols),
    )


def build_model(name: str, seed: int):
    if name == "et":
        return ExtraTreesClassifier(
            n_estimators=500,
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=4,
        )
    if name == "mlp":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="constant", fill_value=-999.0)),
                ("scaler", StandardScaler()),
                (
                    "mlp",
                    MLPClassifier(
                        hidden_layer_sizes=(256, 128, 64),
                        activation="relu",
                        alpha=3e-4,
                        batch_size=2048,
                        learning_rate_init=8e-4,
                        max_iter=60,
                        early_stopping=True,
                        validation_fraction=0.1,
                        n_iter_no_change=8,
                        random_state=seed,
                        verbose=False,
                    ),
                ),
            ]
        )
    if name == "logreg":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="constant", fill_value=-999.0)),
                ("scaler", StandardScaler()),
                ("logreg", LogisticRegression(max_iter=600, C=0.5)),
            ]
        )
    raise ValueError(f"Unsupported model: {name}")


def predict_proba(model, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


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

    model_names = [name.strip() for name in args.models.split(",") if name.strip()]
    y = train_df[TARGET].astype(int)
    splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)

    oof_by_model: dict[str, np.ndarray] = {name: np.zeros(len(train_df), dtype=float) for name in model_names}
    test_by_model: dict[str, np.ndarray] = {name: np.zeros(len(test_df), dtype=float) for name in model_names}

    for fold_idx, (train_idx, valid_idx) in enumerate(splitter.split(train_df, y), start=1):
        fold_train = train_df.iloc[train_idx].copy()
        fold_valid = train_df.iloc[valid_idx].copy()
        fold_test = test_df.copy()

        if external_df is not None and args.external_frac > 0:
            sampled_external = external_df.sample(
                frac=min(args.external_frac, 1.0),
                random_state=args.seed + fold_idx,
            )
            fold_train = pd.concat([fold_train, sampled_external], axis=0, ignore_index=True)

        fold_train, fold_valid, fold_test = add_fold_encodings(fold_train, fold_valid, fold_test)
        train_x, valid_x, test_x = to_dense_matrix(fold_train, fold_valid, fold_test)
        train_x, valid_x, test_x = correlation_filter(train_x, valid_x, test_x, args.corr_threshold)

        train_y = fold_train[TARGET].astype(int)
        valid_y = fold_valid[TARGET].astype(int)

        for model_name in model_names:
            model = build_model(model_name, args.seed + fold_idx)
            model.fit(train_x, train_y)
            valid_pred = predict_proba(model, valid_x)
            test_pred = predict_proba(model, test_x)

            oof_by_model[model_name][valid_idx] = valid_pred
            test_by_model[model_name] += test_pred / args.folds

            fold_auc = roc_auc_score(valid_y, valid_pred)
            print(f"[{model_name}] fold {fold_idx}/{args.folds} auc={fold_auc:.6f}")

    blend_oof = np.mean(np.column_stack([oof_by_model[name] for name in model_names]), axis=1)
    blend_test = np.mean(np.column_stack([test_by_model[name] for name in model_names]), axis=1)

    for model_name in model_names:
        model_auc = roc_auc_score(y, oof_by_model[model_name])
        print(f"[{model_name}] overall oof auc={model_auc:.6f}")
        model_output = args.output_dir / f"submission_{model_name}.csv"
        pd.DataFrame({ID_COL: test_raw[ID_COL], TARGET: test_by_model[model_name]}).to_csv(model_output, index=False)

    blend_auc = roc_auc_score(y, blend_oof)
    print(f"[blend] overall oof auc={blend_auc:.6f}")

    pd.DataFrame(
        {
            ID_COL: train_raw[ID_COL],
            TARGET: train_raw[TARGET],
            "prediction": blend_oof,
        }
    ).to_csv(args.output_dir / "oof_predictions_blend.csv", index=False)

    if args.sample_submission_path and args.sample_submission_path.exists():
        submission = pd.read_csv(args.sample_submission_path)
        submission[TARGET] = blend_test
    else:
        submission = pd.DataFrame({ID_COL: test_raw[ID_COL], TARGET: blend_test})

    submission.to_csv(args.output_dir / "submission_blend.csv", index=False)


if __name__ == "__main__":
    main()
