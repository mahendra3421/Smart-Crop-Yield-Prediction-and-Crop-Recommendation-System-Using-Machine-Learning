"""
models/hyperparameter_tuning.py
--------------------------------
GridSearchCV, RandomizedSearchCV, and Bayesian Optimization (Optuna)
for the best-performing models.
"""

import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, KFold
from sklearn.ensemble        import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics         import make_scorer, mean_squared_error


def _rmse_scorer():
    def rmse(y_true, y_pred):
        return -np.sqrt(mean_squared_error(y_true, y_pred))
    return make_scorer(rmse)


CV = KFold(n_splits=5, shuffle=True, random_state=42)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Grid Search
# ─────────────────────────────────────────────────────────────────────────────

def grid_search_rf(X_train, y_train) -> tuple:
    """GridSearchCV on Random Forest."""
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth":    [None, 10, 20],
        "max_features": ["sqrt", "log2"],
        "min_samples_split": [2, 5],
    }
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    gs = GridSearchCV(rf, param_grid, cv=CV, scoring=_rmse_scorer(),
                      n_jobs=-1, verbose=0, refit=True)
    gs.fit(X_train, y_train)
    print(f"[GridSearch RF] Best params: {gs.best_params_}")
    return gs.best_estimator_, gs.best_params_


# ─────────────────────────────────────────────────────────────────────────────
# 2. Randomized Search
# ─────────────────────────────────────────────────────────────────────────────

def randomized_search_xgb(X_train, y_train, n_iter: int = 30) -> tuple:
    """RandomizedSearchCV on XGBoost."""
    try:
        from xgboost import XGBRegressor
    except ImportError:
        print("[RandomSearch] XGBoost not installed – skipping.")
        return None, {}

    param_dist = {
        "n_estimators":    [100, 200, 300, 500],
        "max_depth":       [3, 4, 5, 6, 8],
        "learning_rate":   [0.01, 0.03, 0.05, 0.1, 0.2],
        "subsample":       [0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree":[0.6, 0.7, 0.8, 0.9, 1.0],
        "reg_alpha":       [0, 0.01, 0.1, 0.5, 1.0],
        "reg_lambda":      [0.5, 1.0, 2.0, 5.0],
    }
    xgb = XGBRegressor(random_state=42, n_jobs=-1, verbosity=0)
    rs  = RandomizedSearchCV(xgb, param_dist, n_iter=n_iter,
                              cv=CV, scoring=_rmse_scorer(),
                              n_jobs=-1, verbose=0, refit=True,
                              random_state=42)
    rs.fit(X_train, y_train)
    print(f"[RandomSearch XGB] Best params: {rs.best_params_}")
    return rs.best_estimator_, rs.best_params_


# ─────────────────────────────────────────────────────────────────────────────
# 3. Bayesian Optimization – Optuna
# ─────────────────────────────────────────────────────────────────────────────

def bayesian_optimize_lgbm(X_train, y_train,
                            n_trials: int = 40) -> tuple:
    """
    Bayesian HPO for LightGBM using Optuna.
    Returns the fitted best model and best params dict.
    """
    try:
        import optuna
        from lightgbm import LGBMRegressor
    except ImportError:
        print("[Optuna] optuna / lightgbm not installed – skipping Bayesian HPO.")
        return None, {}

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            "n_estimators":    trial.suggest_int("n_estimators", 100, 500),
            "max_depth":       trial.suggest_int("max_depth", 3, 12),
            "learning_rate":   trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves":      trial.suggest_int("num_leaves", 20, 300),
            "subsample":       trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree":trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha":       trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda":      trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        }
        model = LGBMRegressor(**params, random_state=42, n_jobs=-1, verbose=-1)
        from sklearn.model_selection import cross_val_score
        scores = cross_val_score(model, X_train, y_train,
                                 cv=CV, scoring=_rmse_scorer(), n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    print(f"[Optuna LightGBM] Best params: {best_params}")

    from lightgbm import LGBMRegressor
    best_model = LGBMRegressor(**best_params, random_state=42, n_jobs=-1, verbose=-1)
    best_model.fit(X_train, y_train)

    return best_model, best_params


# ─────────────────────────────────────────────────────────────────────────────
# 4. Run all tuning
# ─────────────────────────────────────────────────────────────────────────────

def tune_all_models(X_train, y_train,
                    run_grid: bool = True,
                    run_random: bool = True,
                    run_bayesian: bool = True,
                    n_trials: int = 30) -> dict:
    """
    Run all three HPO methods and return a dict of tuned models.
    """
    tuned = {}

    if run_grid:
        print("\n[HPO] → GridSearchCV  (RandomForest)")
        model, params = grid_search_rf(X_train, y_train)
        if model is not None:
            tuned["RF_GridSearch"] = {"model": model, "params": params}

    if run_random:
        print("\n[HPO] → RandomizedSearchCV  (XGBoost)")
        model, params = randomized_search_xgb(X_train, y_train, n_iter=n_trials)
        if model is not None:
            tuned["XGB_RandomSearch"] = {"model": model, "params": params}

    if run_bayesian:
        print("\n[HPO] → Bayesian Optimization  (LightGBM via Optuna)")
        model, params = bayesian_optimize_lgbm(X_train, y_train, n_trials=n_trials)
        if model is not None:
            tuned["LGBM_Bayesian"] = {"model": model, "params": params}

    return tuned
