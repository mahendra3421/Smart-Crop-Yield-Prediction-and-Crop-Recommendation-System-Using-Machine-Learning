"""
preprocessing/feature_engineering.py
---------------------------------------
Advanced feature selection:
- Principal Component Analysis (PCA)
- Recursive Feature Elimination (RFE)
- Mutual Information
- SHAP-based feature importance
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.decomposition    import PCA
from sklearn.feature_selection import RFE, mutual_info_regression, mutual_info_classif
from sklearn.ensemble         import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing    import StandardScaler


# ─────────────────────────────────────────────────────────────────────────────
# 1. PCA
# ─────────────────────────────────────────────────────────────────────────────

def apply_pca(X_train: pd.DataFrame, X_test: pd.DataFrame,
              n_components: float = 0.95,
              return_pca: bool = True):
    """
    Fit PCA on training data and transform both splits.

    Parameters
    ----------
    n_components : float < 1  → keep that fraction of variance
                   int        → keep that many components
    """
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_te_s = scaler.transform(X_test)

    pca = PCA(n_components=n_components, random_state=42)
    X_tr_pca = pca.fit_transform(X_tr_s)
    X_te_pca = pca.transform(X_te_s)

    n_comp = pca.n_components_
    var    = pca.explained_variance_ratio_.cumsum()[-1]
    print(f"[PCA] {n_comp} components explain {var*100:.1f}% variance.")

    cols = [f"PC{i+1}" for i in range(n_comp)]
    X_tr_df = pd.DataFrame(X_tr_pca, columns=cols, index=X_train.index)
    X_te_df = pd.DataFrame(X_te_pca, columns=cols, index=X_test.index)

    if return_pca:
        return X_tr_df, X_te_df, pca
    return X_tr_df, X_te_df


# ─────────────────────────────────────────────────────────────────────────────
# 2. RECURSIVE FEATURE ELIMINATION (RFE)
# ─────────────────────────────────────────────────────────────────────────────

def apply_rfe(X_train: pd.DataFrame, y_train: pd.Series,
              n_features: int = 10,
              estimator=None) -> tuple[pd.Index, object]:
    """
    RFE with a GradientBoostingRegressor as the base estimator.
    Returns selected feature names and the fitted RFE object.
    """
    if estimator is None:
        estimator = GradientBoostingRegressor(n_estimators=100, random_state=42)

    rfe = RFE(estimator=estimator, n_features_to_select=n_features, step=1)
    rfe.fit(X_train, y_train)

    selected = X_train.columns[rfe.support_]
    print(f"[RFE] Selected {len(selected)} features: {selected.tolist()}")
    return selected, rfe


# ─────────────────────────────────────────────────────────────────────────────
# 3. MUTUAL INFORMATION
# ─────────────────────────────────────────────────────────────────────────────

def mutual_information_ranking(X_train: pd.DataFrame, y_train: pd.Series,
                                task: str = "regression",
                                top_k: int = 10) -> pd.DataFrame:
    """
    Compute mutual information between each feature and the target.
    task : 'regression' | 'classification'
    Returns a DataFrame ranked by MI score.
    """
    if task == "regression":
        mi = mutual_info_regression(X_train, y_train, random_state=42)
    else:
        mi = mutual_info_classif(X_train, y_train, random_state=42)

    mi_df = pd.DataFrame({
        "feature": X_train.columns,
        "mi_score": mi,
    }).sort_values("mi_score", ascending=False).head(top_k)

    print("[MI] Top features by mutual information:")
    print(mi_df.to_string(index=False))
    return mi_df


# ─────────────────────────────────────────────────────────────────────────────
# 4. SHAP FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────────────────────

def shap_feature_importance(model, X_train: pd.DataFrame,
                             max_display: int = 15,
                             save_path: str = None) -> pd.DataFrame:
    """
    Compute SHAP values using TreeExplainer (for tree models) or
    KernelExplainer (fallback), and plot a summary bar chart.
    """
    try:
        import shap
    except ImportError:
        print("[SHAP] 'shap' package not installed – skipping.")
        return pd.DataFrame()

    try:
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_train)
    except Exception:
        print("[SHAP] TreeExplainer failed – using KernelExplainer on a sample.")
        sample  = X_train.sample(min(100, len(X_train)), random_state=42)
        explainer = shap.KernelExplainer(model.predict, sample)
        shap_vals = explainer.shap_values(sample)
        X_train = sample

    if isinstance(shap_vals, list):
        # multi-output → take first
        shap_vals = shap_vals[0]

    mean_abs_shap = np.abs(shap_vals).mean(axis=0)
    shap_df = pd.DataFrame({
        "feature":    X_train.columns,
        "shap_value": mean_abs_shap,
    }).sort_values("shap_value", ascending=False)

    # Plot
    top = shap_df.head(max_display)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top["feature"][::-1], top["shap_value"][::-1], color="#4E79A7")
    ax.set_xlabel("Mean |SHAP|")
    ax.set_title("SHAP Feature Importance")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    return shap_df


# ─────────────────────────────────────────────────────────────────────────────
# 5. COMBINED SELECTOR
# ─────────────────────────────────────────────────────────────────────────────

def select_best_features(X_train: pd.DataFrame, y_train: pd.Series,
                          method: str = "mi",
                          n_features: int = 12) -> list[str]:
    """
    Convenience wrapper: pick top features by 'mi', 'rfe', or 'rf_importance'.
    Returns a list of column names.
    """
    if method == "mi":
        mi_df = mutual_information_ranking(X_train, y_train, top_k=n_features)
        return mi_df["feature"].tolist()

    elif method == "rfe":
        selected, _ = apply_rfe(X_train, y_train, n_features=n_features)
        return selected.tolist()

    elif method == "rf_importance":
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        imp_df = pd.DataFrame({
            "feature":    X_train.columns,
            "importance": rf.feature_importances_,
        }).sort_values("importance", ascending=False).head(n_features)
        print("[RF] Top features by importance:")
        print(imp_df.to_string(index=False))
        return imp_df["feature"].tolist()

    else:
        raise ValueError(f"Unknown method: {method}. Choose 'mi', 'rfe', or 'rf_importance'.")
