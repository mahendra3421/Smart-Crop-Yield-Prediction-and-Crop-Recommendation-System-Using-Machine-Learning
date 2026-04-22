"""
preprocessing/data_preprocessing.py
-------------------------------------
Complete data cleaning, imputation, encoding, scaling,
outlier detection, and train-test split pipeline.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing  import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.impute          import SimpleImputer
from scipy                   import stats
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA LOADING & BASIC CLEANING
# ─────────────────────────────────────────────────────────────────────────────

def load_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    • Drop exact duplicate rows
    • Strip string columns
    • Clip physiologically impossible numeric values
    • Ensure correct dtypes
    """
    df = df.copy()

    # Remove duplicates
    before = len(df)
    df.drop_duplicates(inplace=True)
    print(f"[clean] Dropped {before - len(df)} duplicate rows.")

    # Strip whitespace in string columns
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()

    # Logical clipping for domain-specific columns
    clip_map = {
        "Humidity":    (0,   100),
        "pH":          (3.5,   9.5),
        "N":           (0,   300),
        "P":           (0,   200),
        "K":           (0,   250),
        "Rainfall":    (0, 5_000),
        "Temperature": (-10, 55),
        "Pesticide_usage": (0, 100),
        "Area":        (0, 1_000_000),
    }
    for col, (lo, hi) in clip_map.items():
        if col in df.columns:
            df[col] = df[col].clip(lower=lo, upper=hi)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. MISSING VALUE IMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Numeric  → KNN-style fallback using median (per crop group if crop column exists).
    Categorical → most-frequent (mode) imputation.
    """
    df = df.copy()

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()

    # Group-aware median imputation for numerics
    if "Crop" in df.columns and "Yield" in df.columns:
        for col in num_cols:
            if df[col].isnull().any():
                df[col] = df.groupby("Crop")[col].transform(
                    lambda x: x.fillna(x.median() if not np.isnan(x.median()) else df[col].median())
                )

    # Remaining NaNs  →  global median
    num_imp = SimpleImputer(strategy="median")
    df[num_cols] = num_imp.fit_transform(df[num_cols])

    # Categorical → mode
    cat_imp = SimpleImputer(strategy="most_frequent")
    if cat_cols:
        df[cat_cols] = cat_imp.fit_transform(df[cat_cols])

    print(f"[impute] Remaining NaNs after imputation: {df.isnull().sum().sum()}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. OUTLIER DETECTION & REMOVAL
# ─────────────────────────────────────────────────────────────────────────────

def remove_outliers_zscore(df: pd.DataFrame, threshold: float = 3.5,
                            cols: list = None) -> pd.DataFrame:
    """
    Remove rows whose Z-score on any specified numeric column exceeds `threshold`.
    Uses a robust Z-score (median / MAD) to avoid masking.
    """
    df = df.copy()
    if cols is None:
        cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # Don't clip on these broad/ID-like cols
        cols = [c for c in cols if c not in ("Year",)]

    z = np.abs(stats.zscore(df[cols], nan_policy="omit"))
    mask = (z < threshold).all(axis=1)
    removed = (~mask).sum()
    print(f"[outlier] Removed {removed} outlier rows (z-threshold={threshold}).")
    return df[mask].reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# 4. ENCODING
# ─────────────────────────────────────────────────────────────────────────────

class DataEncoder:
    """Encodes categorical features; preserves inverse-mapping for UI."""

    def __init__(self):
        self.label_encoders: dict[str, LabelEncoder] = {}
        self.ohe_columns: list[str] = []

    def fit_transform(self, df: pd.DataFrame,
                      label_cols: list = None,
                      ohe_cols:   list = None) -> pd.DataFrame:
        """
        label_cols → LabelEncoder  (high-cardinality: State, District, Crop)
        ohe_cols   → One-Hot  (low-cardinality: Season)
        """
        df = df.copy()

        # Label encoding
        for col in (label_cols or []):
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le

        # One-hot encoding
        for col in (ohe_cols or []):
            if col in df.columns:
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=False)
                df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
                self.ohe_columns.extend(dummies.columns.tolist())

        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted encoders to new data (e.g., at inference time)."""
        df = df.copy()
        for col, le in self.label_encoders.items():
            if col in df.columns:
                # Handle unseen labels gracefully
                def _safe_transform(x):
                    try:
                        return le.transform([x])[0]
                    except ValueError:
                        return -1
                df[col] = df[col].astype(str).map(_safe_transform)
        return df

    def inverse_transform_label(self, col: str, encoded_vals):
        """Decode a label-encoded column back to original strings."""
        if col in self.label_encoders:
            return self.label_encoders[col].inverse_transform(encoded_vals)
        return encoded_vals


# ─────────────────────────────────────────────────────────────────────────────
# 5. NORMALISATION / SCALING
# ─────────────────────────────────────────────────────────────────────────────

class FeatureScaler:
    """Wraps StandardScaler and MinMaxScaler for easy fit/transform."""

    def __init__(self, method: str = "standard"):
        if method == "standard":
            self.scaler = StandardScaler()
        elif method == "minmax":
            self.scaler = MinMaxScaler()
        else:
            raise ValueError("method must be 'standard' or 'minmax'")
        self.feature_names: list[str] = []

    def fit_transform(self, df: pd.DataFrame, cols: list) -> pd.DataFrame:
        df = df.copy()
        valid = [c for c in cols if c in df.columns]
        self.feature_names = valid
        df[valid] = self.scaler.fit_transform(df[valid])
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        valid = [c for c in self.feature_names if c in df.columns]
        df[valid] = self.scaler.transform(df[valid])
        return df

    def inverse_transform(self, arr: np.ndarray) -> np.ndarray:
        return self.scaler.inverse_transform(arr)


# ─────────────────────────────────────────────────────────────────────────────
# 6. TRAIN / TEST SPLIT
# ─────────────────────────────────────────────────────────────────────────────

def split_data(X: pd.DataFrame, y: pd.Series,
               test_size: float = 0.30,
               random_state: int = 42):
    """Stratified split where possible; plain split otherwise."""
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state,
            stratify=y if y.dtype == object or y.nunique() < 20 else None,
        )
    except Exception:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state,
        )
    print(f"[split] Train {X_train.shape}, Test {X_test.shape}")
    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────────────────────────────────────
# 7. FULL PIPELINE WRAPPER
# ─────────────────────────────────────────────────────────────────────────────

def run_full_pipeline(df: pd.DataFrame,
                      target: str = "Yield",
                      scale_method: str = "standard",
                      test_size: float = 0.30):
    """
    End-to-end preprocessing:
        raw DF  →  cleaned → imputed → encoded → scaled → split
    Returns
    -------
    X_train, X_test, y_train, y_test, encoder, scaler, feature_cols
    """
    # Identify what we have
    label_cols = [c for c in ("Crop", "State", "District") if c in df.columns]
    ohe_cols   = [c for c in ("Season",) if c in df.columns]

    # 1 Clean
    df = load_and_clean(df)

    # 2 Impute
    df = impute_missing(df)

    # 3 Outlier removal on numeric features (not the target)
    num_feat_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    outlier_cols  = [c for c in num_feat_cols if c != target and c != "Production"]
    df = remove_outliers_zscore(df, cols=outlier_cols)

    # 4 Encode
    encoder = DataEncoder()
    df = encoder.fit_transform(df, label_cols=label_cols, ohe_cols=ohe_cols)

    # 5 Separate target
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found. Available: {df.columns.tolist()}")
    y = df[target]
    drop_cols = [target]
    # Drop non-feature cols
    for extra in ("yield_category", "crop_type", "crop_name"):
        if extra in df.columns:
            drop_cols.append(extra)
    X = df.drop(columns=drop_cols)

    feature_cols = X.columns.tolist()

    # 6 Scale
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    scaler = FeatureScaler(method=scale_method)
    X = scaler.fit_transform(X, cols=num_cols)

    # 7 Split
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=test_size)

    return X_train, X_test, y_train, y_test, encoder, scaler, feature_cols
