"""
models/model_evaluation.py
----------------------------
Comprehensive model evaluation metrics and comparison utilities.
Covers: RMSE, MAE, R², MAPE + precision/recall/F1 for classification tasks.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
)


# ─────────────────────────────────────────────────────────────────────────────
# Regression metrics
# ─────────────────────────────────────────────────────────────────────────────

def regression_metrics(y_true, y_pred, model_name: str = "Model") -> dict:
    """Return a dict of regression evaluation metrics."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100)
    nrmse = rmse / (y_true.max() - y_true.min() + 1e-9)

    metrics = {
        "Model":  model_name,
        "RMSE":   round(rmse, 4),
        "MAE":    round(mae,  4),
        "R²":     round(r2,   4),
        "MAPE%":  round(mape, 2),
        "NRMSE":  round(nrmse, 4),
    }
    return metrics


def evaluate_all_models(models: dict, X_test, y_test) -> pd.DataFrame:
    """
    Evaluate all models and return a comparison DataFrame.

    Parameters
    ----------
    models : { name: fitted_estimator }
    """
    results = []
    for name, model in models.items():
        try:
            y_pred = model.predict(X_test)
            m = regression_metrics(y_test, y_pred, model_name=name)
            results.append(m)
            print(f"  {name:<25} RMSE={m['RMSE']:.4f}  MAE={m['MAE']:.4f}  R²={m['R²']:.4f}")
        except Exception as e:
            print(f"  [WARN] {name}: {e}")

    df = pd.DataFrame(results).sort_values("R²", ascending=False).reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Classification metrics (for crop recommendation model)
# ─────────────────────────────────────────────────────────────────────────────

def classification_metrics(y_true, y_pred, model_name: str = "Model") -> dict:
    acc   = accuracy_score(y_true, y_pred)
    prec  = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec   = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1    = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    return {
        "Model":     model_name,
        "Accuracy":  round(acc,  4),
        "Precision": round(prec, 4),
        "Recall":    round(rec,  4),
        "F1":        round(f1,   4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation helpers
# ─────────────────────────────────────────────────────────────────────────────

def plot_model_comparison(results_df: pd.DataFrame, save_path: str = None):
    """Bar chart comparing models across RMSE, MAE, R²."""
    metrics = ["RMSE", "MAE", "R²"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Model Performance Comparison", fontsize=16, fontweight="bold")

    colors = sns.color_palette("husl", len(results_df))

    for ax, metric in zip(axes, metrics):
        vals = results_df[metric].values
        bars = ax.barh(results_df["Model"], vals, color=colors)
        ax.set_xlabel(metric)
        ax.set_title(f"{metric} by Model")
        ax.invert_yaxis()
        for bar, val in zip(bars, vals):
            ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}", va="center", fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return fig


def plot_actual_vs_predicted(y_true, y_pred, model_name: str = "Best Model",
                              save_path: str = None):
    """Scatter plot: actual vs predicted yield."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_true, y_pred, alpha=0.4, color="#4E79A7", edgecolors="white", s=40)
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], "r--", lw=2, label="Perfect prediction")
    ax.set_xlabel("Actual Yield")
    ax.set_ylabel("Predicted Yield")
    ax.set_title(f"Actual vs Predicted — {model_name}")
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return fig


def plot_residuals(y_true, y_pred, model_name: str = "Best Model",
                   save_path: str = None):
    """Residual distribution plot."""
    residuals = np.asarray(y_true) - np.asarray(y_pred)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].scatter(y_pred, residuals, alpha=0.4, color="#E15759", s=30)
    axes[0].axhline(0, color="black", lw=1.5)
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Residual")
    axes[0].set_title("Residuals vs Predicted")

    axes[1].hist(residuals, bins=40, color="#76B7B2", edgecolor="white")
    axes[1].set_xlabel("Residual")
    axes[1].set_title("Residual Distribution")

    fig.suptitle(f"Residual Analysis — {model_name}", fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return fig


def plot_feature_importance(importances: np.ndarray, feature_names: list,
                             top_n: int = 20, title: str = "Feature Importance",
                             save_path: str = None):
    """Horizontal bar chart of feature importances."""
    idx  = np.argsort(importances)[-top_n:]
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(np.array(feature_names)[idx], importances[idx],
            color=sns.color_palette("viridis", top_n))
    ax.set_xlabel("Importance Score")
    ax.set_title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return fig
