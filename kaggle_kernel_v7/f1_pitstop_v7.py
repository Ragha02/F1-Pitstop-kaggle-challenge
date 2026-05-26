
import os
import glob
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

TARGET = 'PitNextLap'
ID_COL = 'id'
FOLDS = 5
SEEDS = [42, 2025]
EXTERNAL_WEIGHT = 0.35

HIGH_CARD_COLS = [
    'Driver', 'Race', 'RaceYear', 'DriverCompound', 'DriverRace', 'CompoundStint',
]
SMALL_ONEHOT_COLS = ['Compound', 'Year', 'PitStop', 'Stint']
NUMERIC_COLS = [
    'LapNumber', 'TyreLife', 'Position', 'LapTime (s)', 'LapTime_Delta',
    'Cumulative_Degradation', 'RaceProgress', 'Position_Change',
    'EstimatedTotalLaps', 'LapsRemaining', 'TyreLifeRatio', 'TyreVsLapRatio',
    'DegradationPerLap', 'DeltaPerLap', 'PositionPct', 'StintTyreLoad',
    'LateRaceTyreLoad', 'TyreRemainingLoad', 'DeltaPositionInteraction',
    'LapTyreInteraction', 'ProgressSquared', 'TyreLifeSquared',
]
BASE_MODELS = ['catboost', 'xgboost', 'histgb', 'lightgbm', 'et']
VOTE_MODELS = ['catboost', 'xgboost', 'histgb', 'lightgbm']

train_matches = glob.glob('/kaggle/input/**/train.csv', recursive=True)
test_matches = glob.glob('/kaggle/input/**/test.csv', recursive=True)
sub_matches = glob.glob('/kaggle/input/**/sample_submission.csv', recursive=True)
ext_matches = glob.glob('/kaggle/input/**/f1_strategy_dataset_v4.csv', recursive=True)

if not train_matches or not test_matches or not sub_matches:
    raise FileNotFoundError(f'Competition files not found under /kaggle/input. Available dirs: {os.listdir("/kaggle/input")}')
if not ext_matches:
    raise FileNotFoundError(f'External dataset file not found under /kaggle/input. Available dirs: {os.listdir("/kaggle/input")}')

TRAIN_PATH = train_matches[0]
TEST_PATH = test_matches[0]
SUB_PATH = sub_matches[0]
EXT_PATH = ext_matches[0]

train_raw = pd.read_csv(TRAIN_PATH)
test_raw = pd.read_csv(TEST_PATH)
sample_sub = pd.read_csv(SUB_PATH)
external_raw = pd.read_csv(EXT_PATH)
external_raw = external_raw.drop(columns=['Normalized_TyreLife'])

print('Train path:', TRAIN_PATH)
print('Test path:', TEST_PATH)
print('External path:', EXT_PATH)
print('Train:', train_raw.shape, 'Test:', test_raw.shape, 'External:', external_raw.shape)


def engineer_features(df):
    df = df.copy()
    df['RaceYear'] = df['Race'] + '|' + df['Year'].astype(str)
    df['DriverCompound'] = df['Driver'] + '|' + df['Compound']
    df['DriverRace'] = df['Driver'] + '|' + df['Race']
    df['CompoundStint'] = df['Compound'] + '|' + df['Stint'].astype(str)
    df['EstimatedTotalLapsRaw'] = (df['LapNumber'] / df['RaceProgress'].replace(0, np.nan)).clip(20, 90)
    race_year_totals = df.groupby('RaceYear')['EstimatedTotalLapsRaw'].transform('median')
    df['EstimatedTotalLaps'] = race_year_totals.fillna(df['EstimatedTotalLapsRaw'])
    df['LapsRemaining'] = (df['EstimatedTotalLaps'] - df['LapNumber']).clip(lower=0)
    df['TyreLifeRatio'] = df['TyreLife'] / df['EstimatedTotalLaps'].replace(0, np.nan)
    df['TyreVsLapRatio'] = df['TyreLife'] / df['LapNumber'].replace(0, np.nan)
    df['DegradationPerLap'] = df['Cumulative_Degradation'] / df['TyreLife'].replace(0, np.nan)
    df['DeltaPerLap'] = df['LapTime_Delta'] / df['TyreLife'].replace(0, np.nan)
    df['PositionPct'] = df['Position'] / 20.0
    df['StintTyreLoad'] = df['Stint'] * df['TyreLife']
    df['LateRaceTyreLoad'] = df['TyreLife'] * df['RaceProgress']
    df['TyreRemainingLoad'] = df['TyreLife'] * df['LapsRemaining']
    df['DeltaPositionInteraction'] = df['LapTime_Delta'] * df['PositionPct']
    df['LapTyreInteraction'] = df['LapNumber'] * df['TyreLife']
    df['ProgressSquared'] = df['RaceProgress'] ** 2
    df['TyreLifeSquared'] = df['TyreLife'] ** 2
    return df.replace([np.inf, -np.inf], np.nan)


def add_fold_encodings(train_df, valid_df, test_df):
    train_df = train_df.copy()
    valid_df = valid_df.copy()
    test_df = test_df.copy()
    global_mean = train_df[TARGET].mean()
    for col in HIGH_CARD_COLS:
        stats = train_df.groupby(col)[TARGET].agg(['sum', 'count'])
        smooth = (stats['sum'] + 20 * global_mean) / (stats['count'] + 20)
        freq = train_df[col].value_counts(normalize=True)
        for df in (train_df, valid_df, test_df):
            df[f'TE_{col}'] = df[col].map(smooth).fillna(global_mean)
            df[f'FE_{col}'] = df[col].map(freq).fillna(0.0)
    return train_df, valid_df, test_df


def to_dense_matrix(train_df, valid_df, test_df):
    dense_cols = NUMERIC_COLS + [f'TE_{col}' for col in HIGH_CARD_COLS] + [f'FE_{col}' for col in HIGH_CARD_COLS]
    def onehot(df):
        return pd.get_dummies(df[SMALL_ONEHOT_COLS].astype(str), prefix=SMALL_ONEHOT_COLS, dtype=float)
    train_small = onehot(train_df)
    valid_small = onehot(valid_df)
    test_small = onehot(test_df)
    train_small, valid_small = train_small.align(valid_small, join='outer', axis=1, fill_value=0.0)
    train_small, test_small = train_small.align(test_small, join='outer', axis=1, fill_value=0.0)
    valid_small, test_small = valid_small.align(test_small, join='outer', axis=1, fill_value=0.0)
    train_x = pd.concat([train_df[dense_cols].reset_index(drop=True), train_small.reset_index(drop=True)], axis=1)
    valid_x = pd.concat([valid_df[dense_cols].reset_index(drop=True), valid_small.reset_index(drop=True)], axis=1)
    test_x = pd.concat([test_df[dense_cols].reset_index(drop=True), test_small.reset_index(drop=True)], axis=1)
    return train_x, valid_x, test_x


def make_base_model(name, seed):
    if name == 'catboost':
        return CatBoostClassifier(iterations=500, depth=8, learning_rate=0.05, l2_leaf_reg=4.0, loss_function='Logloss', eval_metric='AUC', verbose=False, random_seed=seed)
    if name == 'xgboost':
        return XGBClassifier(n_estimators=350, max_depth=8, learning_rate=0.05, subsample=0.9, colsample_bytree=0.82, min_child_weight=4, reg_lambda=3.0, gamma=0.1, objective='binary:logistic', eval_metric='auc', tree_method='hist', random_state=seed, n_jobs=4)
    if name == 'lightgbm':
        return LGBMClassifier(n_estimators=450, learning_rate=0.04, num_leaves=63, min_child_samples=30, subsample=0.9, colsample_bytree=0.85, reg_lambda=2.0, random_state=seed, n_jobs=4, objective='binary', verbose=-1)
    if name == 'histgb':
        return HistGradientBoostingClassifier(learning_rate=0.05, max_depth=8, max_leaf_nodes=63, min_samples_leaf=30, l2_regularization=0.1, random_state=seed)
    if name == 'et':
        return ExtraTreesClassifier(n_estimators=500, min_samples_leaf=2, random_state=seed, n_jobs=4)
    raise ValueError(name)


def predict_prob(model, X):
    return model.predict_proba(X)[:, 1]


def fit_model(name, model, X, y, w):
    if name in {'catboost', 'xgboost', 'lightgbm', 'histgb'}:
        model.fit(X, y, sample_weight=w)
    else:
        model.fit(X, y)

train_df = engineer_features(train_raw)
test_df = engineer_features(test_raw)
external_df = engineer_features(external_raw)
y = train_df[TARGET].astype(int)

seed_test_predictions = []
seed_oof_scores = []

for seed in SEEDS:
    print('\n' + '=' * 70)
    print('SEED', seed)
    print('=' * 70)
    splitter = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=seed)
    oof_base = {name: np.zeros(len(train_df), dtype=float) for name in BASE_MODELS}
    test_base = {name: np.zeros(len(test_df), dtype=float) for name in BASE_MODELS}

    for fold_idx, (train_idx, valid_idx) in enumerate(splitter.split(train_df, y), start=1):
        fold_train = train_df.iloc[train_idx].copy()
        fold_valid = train_df.iloc[valid_idx].copy()
        fold_test = test_df.copy()
        fold_train['_sample_weight'] = 1.0
        ext = external_df.copy()
        ext['_sample_weight'] = EXTERNAL_WEIGHT
        fold_train = pd.concat([fold_train, ext], axis=0, ignore_index=True)
        fold_train, fold_valid, fold_test = add_fold_encodings(fold_train, fold_valid, fold_test)
        train_x, valid_x, test_x = to_dense_matrix(fold_train, fold_valid, fold_test)
        train_y = fold_train[TARGET].astype(int)
        valid_y = fold_valid[TARGET].astype(int)
        train_w = fold_train['_sample_weight'].astype(float).values

        for model_name in BASE_MODELS:
            model = make_base_model(model_name, seed + fold_idx)
            fit_model(model_name, model, train_x, train_y, train_w)
            valid_pred = predict_prob(model, valid_x)
            test_pred = predict_prob(model, test_x)
            oof_base[model_name][valid_idx] = valid_pred
            test_base[model_name] += test_pred / FOLDS
            fold_auc = roc_auc_score(valid_y, valid_pred)
            print(f'[{model_name}] fold {fold_idx}/{FOLDS} auc={fold_auc:.6f}')

    oof_frame = pd.DataFrame({ID_COL: train_raw[ID_COL], TARGET: y})
    test_frame = pd.DataFrame({ID_COL: test_raw[ID_COL]})
    for model_name in BASE_MODELS:
        oof_frame[model_name] = oof_base[model_name]
        test_frame[model_name] = test_base[model_name]
        print(f'[{model_name}] overall oof auc={roc_auc_score(y, oof_base[model_name]):.6f}')

    oof_frame['vote4'] = oof_frame[VOTE_MODELS].mean(axis=1)
    test_frame['vote4'] = test_frame[VOTE_MODELS].mean(axis=1)

    meta_features = BASE_MODELS + ['vote4']
    meta_model = HistGradientBoostingClassifier(learning_rate=0.03, max_depth=3, max_leaf_nodes=15, min_samples_leaf=50, random_state=seed)
    meta_model.fit(oof_frame[meta_features], y)
    stack_oof = predict_prob(meta_model, oof_frame[meta_features])
    stack_test = predict_prob(meta_model, test_frame[meta_features])
    stack_auc = roc_auc_score(y, stack_oof)
    print(f'[stack_histgb] overall oof auc={stack_auc:.6f}')
    seed_oof_scores.append(stack_auc)
    seed_test_predictions.append(stack_test)

final_pred = np.mean(np.column_stack(seed_test_predictions), axis=1)
print('Seed OOF scores:', seed_oof_scores)
print('Mean OOF:', float(np.mean(seed_oof_scores)))

submission = sample_sub.copy()
submission[TARGET] = final_pred
submission.to_csv('submission.csv', index=False)
print('Saved submission.csv')
print(submission.head())
