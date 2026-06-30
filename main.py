# Stellar Class Kaggle ML-only solution
# Competition: playground-series-s6e6
# Models: LightGBM, CatBoost, XGBoost. These are classical machine learning tree boosting models, not deep learning.
# Output: /kaggle/working/submission.csv on Kaggle, or ./submission.csv locally.

import os
import gc
import warnings
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_sample_weight

warnings.filterwarnings("ignore")

# =========================
# 1. Config
# =========================
SEED = 42
N_SPLITS = 5

# If your computer is slow, set FAST_MODE=True first. For final Kaggle submission, use False.
FAST_MODE = False

# Three model ensemble gives better chance for a higher score.
# If one package fails in your environment, remove it from this list.
USE_MODELS = ["lgbm"]

if FAST_MODE:
    N_SPLITS = 3
    USE_MODELS = ["lgbm"]

TARGET = "class"
ID_COL = "id"
CAT_COLS = ["spectral_type", "galaxy_population", "spec_pop"]

np.random.seed(SEED)

# =========================
# 2. Read data
# =========================
def read_data():
    possible_dirs = [
        Path("/kaggle/input/playground-series-s6e6"),
        Path("."),
        Path("/mnt/data"),
    ]

    for d in possible_dirs:
        train_path = d / "train.csv"
        test_path = d / "test.csv"
        sample_path = d / "sample_submission.csv"
        if train_path.exists() and test_path.exists() and sample_path.exists():
            print(f"Reading csv files from: {d}")
            return pd.read_csv(train_path), pd.read_csv(test_path), pd.read_csv(sample_path)

    zip_candidates = [
        Path("/kaggle/input/playground-series-s6e6.zip"),
        Path("playground-series-s6e6.zip"),
        Path("/mnt/data/playground-series-s6e6.zip"),
    ]
    for zp in zip_candidates:
        if zp.exists():
            print(f"Reading csv files from zip: {zp}")
            with ZipFile(zp) as z:
                return (
                    pd.read_csv(z.open("train.csv")),
                    pd.read_csv(z.open("test.csv")),
                    pd.read_csv(z.open("sample_submission.csv")),
                )

    raise FileNotFoundError("Cannot find train.csv, test.csv, sample_submission.csv, or playground-series-s6e6.zip")

# 强制覆盖原来的 read_data()，自动查找 Kaggle 输入文件
def read_data():
    import os
    from pathlib import Path

    input_dir = Path("/kaggle/input")

    print("正在查找 Kaggle 输入文件：")
    for dirname, _, filenames in os.walk(input_dir):
        for filename in filenames:
            print(Path(dirname) / filename)

    train_path = None
    test_path = None
    sample_path = None

    for p in input_dir.rglob("*.csv"):
        name = p.name.lower()

        if name == "train.csv":
            train_path = p
        elif name == "test.csv":
            test_path = p
        elif name == "sample_submission.csv":
            sample_path = p

    print("找到的 train:", train_path)
    print("找到的 test:", test_path)
    print("找到的 sample_submission:", sample_path)

    if train_path is None or test_path is None or sample_path is None:
        raise FileNotFoundError("没有找到比赛数据文件，请确认右侧 Input 里有 Predicting Stellar Class 比赛数据。")

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    sample_submission = pd.read_csv(sample_path)

    return train, test, sample_submission

train, test, sample_submission = read_data()
print("train:", train.shape, "test:", test.shape)
print(train[TARGET].value_counts())

# =========================
# 3. Feature engineering
# =========================
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    bands = ["u", "g", "r", "i", "z"]

    # Color index features: very useful for stellar classification.
    for a, b in [("u", "g"), ("g", "r"), ("r", "i"), ("i", "z"),
                 ("u", "r"), ("u", "i"), ("u", "z"), ("g", "i"), ("g", "z"), ("r", "z")]:
        df[f"{a}_{b}_diff"] = df[a] - df[b]
        df[f"{a}_{b}_sum"] = df[a] + df[b]

    # Magnitude statistics.
    df["mag_mean"] = df[bands].mean(axis=1)
    df["mag_std"] = df[bands].std(axis=1)
    df["mag_min"] = df[bands].min(axis=1)
    df["mag_max"] = df[bands].max(axis=1)
    df["mag_range"] = df["mag_max"] - df["mag_min"]
    df["mag_median"] = df[bands].median(axis=1)

    # Redshift is extremely informative for GALAXY, QSO, STAR.
    red = df["redshift"].astype(float)
    df["redshift_abs"] = red.abs()
    df["redshift_log1p"] = np.log1p(np.clip(red, 0, None))
    df["redshift_sq"] = red ** 2
    df["redshift_sqrt"] = np.sqrt(np.clip(red, 0, None))
    df["redshift_near_zero"] = (red.abs() < 0.003).astype(np.int8)
    df["redshift_low"] = (red < 0.1).astype(np.int8)
    df["redshift_high"] = (red > 0.8).astype(np.int8)

    for b in bands:
        df[f"{b}_times_redshift"] = df[b] * red
        df[f"{b}_minus_redshift"] = df[b] - red
        df[f"{b}_div_redshift1p"] = df[b] / (red.abs() + 1.0)

    # Sky position encoded cyclically.
    alpha_rad = np.deg2rad(df["alpha"])
    delta_rad = np.deg2rad(df["delta"])
    df["alpha_sin"] = np.sin(alpha_rad)
    df["alpha_cos"] = np.cos(alpha_rad)
    df["delta_sin"] = np.sin(delta_rad)
    df["delta_cos"] = np.cos(delta_rad)
    df["sky_x"] = np.cos(delta_rad) * np.cos(alpha_rad)
    df["sky_y"] = np.cos(delta_rad) * np.sin(alpha_rad)
    df["sky_z"] = np.sin(delta_rad)

    # Categorical interaction.
    df["spec_pop"] = df["spectral_type"].astype(str) + "_" + df["galaxy_population"].astype(str)

    # Reduce memory.
    for col in df.columns:
        if df[col].dtype == "float64":
            df[col] = df[col].astype("float32")

    return df

train_fe = add_features(train)
test_fe = add_features(test)

features = [c for c in train_fe.columns if c not in [ID_COL, TARGET]]

# Label encode target.
le = LabelEncoder()
y = le.fit_transform(train_fe[TARGET])
print("class mapping:", dict(zip(le.classes_, le.transform(le.classes_))))

# For LightGBM and CatBoost, keep categorical columns as category or string.
X_lgb = train_fe[features].copy()
T_lgb = test_fe[features].copy()
for c in CAT_COLS:
    X_lgb[c] = X_lgb[c].astype("category")
    T_lgb[c] = T_lgb[c].astype("category")

X_cat = train_fe[features].copy()
T_cat = test_fe[features].copy()
for c in CAT_COLS:
    X_cat[c] = X_cat[c].astype(str)
    T_cat[c] = T_cat[c].astype(str)

# XGBoost is more stable with integer encoded categories.
X_xgb = train_fe[features].copy()
T_xgb = test_fe[features].copy()
for c in CAT_COLS:
    all_values = pd.concat([X_xgb[c], T_xgb[c]], axis=0).astype(str)
    mapping = {v: i for i, v in enumerate(sorted(all_values.unique()))}
    X_xgb[c] = X_xgb[c].astype(str).map(mapping).astype("int16")
    T_xgb[c] = T_xgb[c].astype(str).map(mapping).astype("int16")

# =========================
# 4. Metric and threshold optimization
# =========================
def predict_with_thresholds(proba: np.ndarray, thresholds=None) -> np.ndarray:
    if thresholds is None:
        return np.argmax(proba, axis=1)
    thresholds = np.asarray(thresholds, dtype=float)
    return np.argmax(proba / thresholds.reshape(1, -1), axis=1)


def score_proba(y_true, proba, thresholds=None):
    pred = predict_with_thresholds(proba, thresholds)
    return balanced_accuracy_score(y_true, pred)


def optimize_thresholds(y_true, proba, seed=SEED):
    # Balanced accuracy rewards recall of each class equally. Threshold tuning can improve hard-label submissions.
    rng = np.random.default_rng(seed)
    n_classes = proba.shape[1]

    best_t = np.ones(n_classes)
    best_score = score_proba(y_true, proba, best_t)

    # Random search in log-threshold space.
    for scale in [0.03, 0.06, 0.10, 0.16, 0.24]:
        for _ in range(300):
            t = np.exp(rng.normal(0, scale, size=n_classes))
            s = score_proba(y_true, proba, t)
            if s > best_score:
                best_score = s
                best_t = t

    # Optional scipy refinement.
    try:
        from scipy.optimize import minimize

        def objective(log_t):
            return -score_proba(y_true, proba, np.exp(log_t))

        res = minimize(
            objective,
            x0=np.log(best_t),
            method="Nelder-Mead",
            options={"maxiter": 250, "xatol": 1e-4, "fatol": 1e-6, "disp": False},
        )
        t = np.exp(res.x)
        s = score_proba(y_true, proba, t)
        if s > best_score:
            best_score = s
            best_t = t
    except Exception as e:
        print("scipy threshold refinement skipped:", repr(e))

    # Normalize only for readability. Prediction is unchanged by multiplying all thresholds by same constant.
    best_t = best_t / np.mean(best_t)
    return best_t, best_score


# =========================
# 5. Model training
# =========================
def train_lgbm(X, T, y, folds):
    import lightgbm as lgb

    oof = np.zeros((len(X), len(le.classes_)), dtype=np.float32)
    test_pred = np.zeros((len(T), len(le.classes_)), dtype=np.float32)

    params = dict(
        objective="multiclass",
        num_class=len(le.classes_),
        n_estimators=2500 if not FAST_MODE else 500,
        learning_rate=0.025,
        num_leaves=96,
        max_depth=-1,
        min_child_samples=45,
        subsample=0.88,
        subsample_freq=1,
        colsample_bytree=0.88,
        reg_alpha=0.15,
        reg_lambda=5.0,
        random_state=SEED,
        n_jobs=-1,
        verbose=-1,
    )

    for fold, (tr_idx, va_idx) in enumerate(folds, 1):
        print(f"\n[LGBM] Fold {fold}/{N_SPLITS}")
        model = lgb.LGBMClassifier(**params)
        sw = compute_sample_weight(class_weight="balanced", y=y[tr_idx])
        model.fit(
            X.iloc[tr_idx], y[tr_idx],
            sample_weight=sw,
            eval_set=[(X.iloc[va_idx], y[va_idx])],
            eval_metric="multi_logloss",
            categorical_feature=CAT_COLS,
            callbacks=[lgb.early_stopping(120, verbose=False), lgb.log_evaluation(300)],
        )
        oof[va_idx] = model.predict_proba(X.iloc[va_idx], num_iteration=model.best_iteration_)
        test_pred += model.predict_proba(T, num_iteration=model.best_iteration_) / N_SPLITS
        print("fold balanced accuracy:", score_proba(y[va_idx], oof[va_idx]))
        del model, sw
        gc.collect()

    print("LGBM OOF balanced accuracy:", score_proba(y, oof))
    return oof, test_pred


def train_cat(X, T, y, folds):
    from catboost import CatBoostClassifier

    oof = np.zeros((len(X), len(le.classes_)), dtype=np.float32)
    test_pred = np.zeros((len(T), len(le.classes_)), dtype=np.float32)

    cat_indices = [X.columns.get_loc(c) for c in CAT_COLS]

    for fold, (tr_idx, va_idx) in enumerate(folds, 1):
        print(f"\n[CAT] Fold {fold}/{N_SPLITS}")
        model = CatBoostClassifier(
            loss_function="MultiClass",
            eval_metric="MultiClass",
            iterations=3500 if not FAST_MODE else 600,
            learning_rate=0.035,
            depth=8,
            l2_leaf_reg=4.0,
            random_seed=SEED + fold,
            bootstrap_type="Bayesian",
            bagging_temperature=0.4,
            random_strength=0.5,
            od_type="Iter",
            od_wait=180,
            allow_writing_files=False,
            verbose=300,
        )
        sw = compute_sample_weight(class_weight="balanced", y=y[tr_idx])
        model.fit(
            X.iloc[tr_idx], y[tr_idx],
            sample_weight=sw,
            eval_set=(X.iloc[va_idx], y[va_idx]),
            cat_features=cat_indices,
            use_best_model=True,
        )
        oof[va_idx] = model.predict_proba(X.iloc[va_idx])
        test_pred += model.predict_proba(T) / N_SPLITS
        print("fold balanced accuracy:", score_proba(y[va_idx], oof[va_idx]))
        del model, sw
        gc.collect()

    print("CAT OOF balanced accuracy:", score_proba(y, oof))
    return oof, test_pred


def train_xgb(X, T, y, folds):
    from xgboost import XGBClassifier

    oof = np.zeros((len(X), len(le.classes_)), dtype=np.float32)
    test_pred = np.zeros((len(T), len(le.classes_)), dtype=np.float32)

    params = dict(
        objective="multi:softprob",
        num_class=len(le.classes_),
        eval_metric="mlogloss",
        n_estimators=2200 if not FAST_MODE else 500,
        learning_rate=0.025,
        max_depth=6,
        min_child_weight=2.0,
        subsample=0.88,
        colsample_bytree=0.88,
        reg_alpha=0.1,
        reg_lambda=7.0,
        tree_method="hist",
        random_state=SEED,
        n_jobs=-1,
    )

    for fold, (tr_idx, va_idx) in enumerate(folds, 1):
        print(f"\n[XGB] Fold {fold}/{N_SPLITS}")
        model = XGBClassifier(**params)
        sw = compute_sample_weight(class_weight="balanced", y=y[tr_idx])
        try:
            model.fit(
                X.iloc[tr_idx], y[tr_idx],
                sample_weight=sw,
                eval_set=[(X.iloc[va_idx], y[va_idx])],
                verbose=300,
                early_stopping_rounds=120,
            )
        except TypeError:
            # Some XGBoost versions removed early_stopping_rounds from fit.
            model.set_params(n_estimators=900 if not FAST_MODE else 300)
            model.fit(
                X.iloc[tr_idx], y[tr_idx],
                sample_weight=sw,
                eval_set=[(X.iloc[va_idx], y[va_idx])],
                verbose=300,
            )
        oof[va_idx] = model.predict_proba(X.iloc[va_idx])
        test_pred += model.predict_proba(T) / N_SPLITS
        print("fold balanced accuracy:", score_proba(y[va_idx], oof[va_idx]))
        del model, sw
        gc.collect()

    print("XGB OOF balanced accuracy:", score_proba(y, oof))
    return oof, test_pred


folds = list(StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED).split(X_lgb, y))

oof_dict = {}
test_dict = {}

if "lgbm" in USE_MODELS:
    oof_dict["lgbm"], test_dict["lgbm"] = train_lgbm(X_lgb, T_lgb, y, folds)

if "cat" in USE_MODELS:
    oof_dict["cat"], test_dict["cat"] = train_cat(X_cat, T_cat, y, folds)

if "xgb" in USE_MODELS:
    oof_dict["xgb"], test_dict["xgb"] = train_xgb(X_xgb, T_xgb, y, folds)

# =========================
# 6. Blend model probabilities
# =========================
def find_best_blend(oof_dict, y_true, seed=SEED):
    names = list(oof_dict.keys())
    if len(names) == 1:
        return {names[0]: 1.0}, oof_dict[names[0]], score_proba(y_true, oof_dict[names[0]])

    rng = np.random.default_rng(seed)
    candidates = []

    # Single model and equal weights.
    for i, name in enumerate(names):
        w = np.zeros(len(names))
        w[i] = 1.0
        candidates.append(w)
    candidates.append(np.ones(len(names)) / len(names))

    # Random Dirichlet search.
    for _ in range(800):
        candidates.append(rng.dirichlet(np.ones(len(names))))

    best_w = None
    best_p = None
    best_score = -1
    for w in candidates:
        p = np.zeros_like(next(iter(oof_dict.values())))
        for wi, name in zip(w, names):
            p += wi * oof_dict[name]
        s = score_proba(y_true, p)
        if s > best_score:
            best_score = s
            best_w = w
            best_p = p

    return dict(zip(names, best_w)), best_p, best_score

blend_weights, oof_blend, blend_score = find_best_blend(oof_dict, y)
print("\nBest blend weights:", blend_weights)
print("Blend OOF balanced accuracy before thresholds:", blend_score)

thresholds, threshold_score = optimize_thresholds(y, oof_blend)
print("Best thresholds:", thresholds)
print("Blend OOF balanced accuracy after thresholds:", threshold_score)

pred_oof = predict_with_thresholds(oof_blend, thresholds)
print("\nClassification report on OOF:")
print(classification_report(y, pred_oof, target_names=le.classes_))
print("Confusion matrix on OOF:")
print(confusion_matrix(y, pred_oof))

# Build final test probability.
test_blend = np.zeros_like(next(iter(test_dict.values())))
for name, w in blend_weights.items():
    test_blend += w * test_dict[name]

test_pred_num = predict_with_thresholds(test_blend, thresholds)
test_pred_label = le.inverse_transform(test_pred_num)

submission = sample_submission.copy()
submission[TARGET] = test_pred_label

out_dir = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")
submission_path = out_dir / "submission.csv"
submission.to_csv(submission_path, index=False)
print("\nSaved:", submission_path)
print(submission.head())
print(submission[TARGET].value_counts())
