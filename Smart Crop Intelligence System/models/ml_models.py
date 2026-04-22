"""
models/ml_models.py
---------------------
All ML model definitions:
  Random Forest, SVM, XGBoost, LightGBM, GradientBoosting, CatBoost, MLP
  + XGBoost+NN Stacking Ensemble
"""

import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble    import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm         import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge

try:
    from xgboost   import XGBRegressor
except ImportError:
    XGBRegressor = None

try:
    from lightgbm  import LGBMRegressor
except ImportError:
    LGBMRegressor = None

try:
    from catboost  import CatBoostRegressor
except ImportError:
    CatBoostRegressor = None


# ─────────────────────────────────────────────────────────────────────────────
# Model factory
# ─────────────────────────────────────────────────────────────────────────────

def get_base_models() -> dict:
    """
    Return a dict of { model_name: sklearn-compatible estimator }.
    Each model uses research-grade default hyper-parameters.
    """
    models = {}

    models["RandomForest"] = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        bootstrap=True,
        n_jobs=-1,
        random_state=42,
    )

    models["SVM"] = SVR(
        kernel="rbf",
        C=10.0,
        gamma="scale",
        epsilon=0.1,
    )

    if XGBRegressor is not None:
        models["XGBoost"] = XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )

    if LGBMRegressor is not None:
        models["LightGBM"] = LGBMRegressor(
            n_estimators=300,
            max_depth=-1,
            learning_rate=0.05,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )

    models["GradientBoosting"] = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.8,
        random_state=42,
    )

    if CatBoostRegressor is not None:
        models["CatBoost"] = CatBoostRegressor(
            iterations=300,
            depth=6,
            learning_rate=0.05,
            l2_leaf_reg=3,
            random_seed=42,
            verbose=0,
        )

    models["MLP"] = MLPRegressor(
        hidden_layer_sizes=(256, 128, 64),
        activation="relu",
        solver="adam",
        learning_rate="adaptive",
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
    )

    return models


# ─────────────────────────────────────────────────────────────────────────────
# Stacking Ensemble  (XGBoost + Ridge meta-learner)
# ─────────────────────────────────────────────────────────────────────────────

class StackingEnsemble:
    """
    Two-level stacking:
      Level-0  →  XGBoost + GradientBoosting + RandomForest
      Level-1  →  Ridge regression meta-learner
    Uses out-of-fold predictions to prevent leakage.
    """

    def __init__(self, n_folds: int = 5):
        from sklearn.model_selection import KFold

        self.n_folds = n_folds
        self.kf      = KFold(n_splits=n_folds, shuffle=True, random_state=42)

        base = get_base_models()
        self.base_names = ["XGBoost", "GradientBoosting", "RandomForest"]
        self.base_models = [base[n] for n in self.base_names
                            if n in base]   # skip missing

        self.meta_model  = Ridge(alpha=1.0)
        self._fitted_base: list = []

    # ── train ──
    def fit(self, X: np.ndarray, y: np.ndarray):
        import copy
        n            = len(X)
        n_base       = len(self.base_models)
        oof_preds    = np.zeros((n, n_base))

        # Out-of-fold predictions for meta training
        for fold_i, (tr_idx, va_idx) in enumerate(self.kf.split(X)):
            X_tr, X_va = X[tr_idx], X[va_idx]
            y_tr        = y[tr_idx]
            for m_i, model in enumerate(self.base_models):
                m = copy.deepcopy(model)
                m.fit(X_tr, y_tr)
                oof_preds[va_idx, m_i] = m.predict(X_va)

        # Fit base models on full training data
        self._fitted_base = []
        for model in self.base_models:
            m = copy.deepcopy(model)
            m.fit(X, y)
            self._fitted_base.append(m)

        # Fit meta-learner
        self.meta_model.fit(oof_preds, y)
        return self

    # ── predict ──
    def predict(self, X: np.ndarray) -> np.ndarray:
        base_preds = np.column_stack(
            [m.predict(X) for m in self._fitted_base]
        )
        return self.meta_model.predict(base_preds)
