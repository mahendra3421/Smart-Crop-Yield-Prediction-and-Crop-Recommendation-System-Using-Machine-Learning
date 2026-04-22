"""
visualization/plots.py
------------------------
All EDA and results visualizations:
  - Correlation heatmap
  - Yield distribution
  - Feature importance (bar)
  - Model comparison (radar & bar)
  - Seasonal analysis
  - Partial dependence plots
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# Unified style
plt.rcParams.update({
    "font.family":    "DejaVu Sans",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":      True,
    "grid.alpha":     0.3,
    "figure.dpi":     120,
})
PALETTE = "husl"


# ─────────────────────────────────────────────────────────────────────────────
# 1. Correlation Heatmap
# ─────────────────────────────────────────────────────────────────────────────

def plot_correlation_heatmap(df: pd.DataFrame, save_path: str = None):
    num_df = df.select_dtypes(include=[np.number])
    corr   = num_df.corr()

    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                cmap="RdYlGn", center=0, linewidths=0.5,
                annot_kws={"size": 8}, ax=ax)
    ax.set_title("Feature Correlation Heatmap", fontsize=16, fontweight="bold", pad=15)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] Correlation heatmap saved → {save_path or 'not saved'}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Yield Distribution
# ─────────────────────────────────────────────────────────────────────────────

def plot_yield_distribution(df: pd.DataFrame, yield_col: str = "Yield",
                             save_path: str = None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram + KDE
    axes[0].hist(df[yield_col].dropna(), bins=50,
                 color="#4E79A7", edgecolor="white", density=True, alpha=0.7)
    df[yield_col].dropna().plot.kde(ax=axes[0], color="red", lw=2)
    axes[0].set_title("Yield Distribution (PDF)", fontweight="bold")
    axes[0].set_xlabel("Yield")

    # By crop (if available)
    if "Crop" in df.columns:
        crop_means = df.groupby("Crop")[yield_col].mean().sort_values(ascending=False)
        colors = sns.color_palette(PALETTE, len(crop_means))
        axes[1].bar(crop_means.index, crop_means.values, color=colors)
        axes[1].set_title("Average Yield by Crop", fontweight="bold")
        axes[1].set_xlabel("Crop")
        axes[1].set_ylabel("Mean Yield")
        plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha="right")
    else:
        df[yield_col].dropna().plot.box(ax=axes[1])
        axes[1].set_title("Yield Box-Plot")

    fig.suptitle("Yield Analysis", fontsize=15, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# 3. Feature Importance (from fitted model)
# ─────────────────────────────────────────────────────────────────────────────

def plot_rf_feature_importance(model, feature_names: list,
                                top_n: int = 20, save_path: str = None):
    importances = model.feature_importances_
    idx  = np.argsort(importances)[-top_n:]
    labels = np.array(feature_names)[idx]
    values = importances[idx]

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = sns.color_palette("viridis", len(idx))
    ax.barh(labels, values, color=colors)
    ax.set_xlabel("Importance", fontsize=12)
    ax.set_title("Random Forest — Top Feature Importances", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# 4. Model Comparison Bar Charts
# ─────────────────────────────────────────────────────────────────────────────

def plot_model_comparison_bars(results_df: pd.DataFrame, save_path: str = None):
    """
    Side-by-side bar charts for RMSE, MAE, R².
    results_df must have columns: Model, RMSE, MAE, R²
    """
    metrics = ["RMSE", "MAE", "R²"]
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(18, 6))
    fig.suptitle("Model Comparison Summary", fontsize=16, fontweight="bold")

    palette = sns.color_palette(PALETTE, len(results_df))
    for ax, metric in zip(axes, metrics):
        sorted_df = results_df.sort_values(metric, ascending=(metric != "R²"))
        bars = ax.barh(sorted_df["Model"], sorted_df[metric], color=palette)
        ax.set_title(metric, fontsize=13, fontweight="bold")
        ax.set_xlabel(metric)
        for bar, val in zip(bars, sorted_df[metric]):
            ax.text(bar.get_width() + max(sorted_df[metric].max() * 0.01, 0.001),
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}", va="center", fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# 5. Seasonal / Crop Heatmap
# ─────────────────────────────────────────────────────────────────────────────

def plot_crop_season_heatmap(df: pd.DataFrame, yield_col: str = "Yield",
                              save_path: str = None):
    if "Crop" not in df.columns or "Season" not in df.columns:
        return

    pivot = df.pivot_table(values=yield_col, index="Crop",
                           columns="Season", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlOrRd",
                linewidths=0.5, ax=ax)
    ax.set_title("Average Yield by Crop × Season", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# 6. Scatter Matrix (Pairplot) – lightweight version
# ─────────────────────────────────────────────────────────────────────────────

def plot_pairplot_sample(df: pd.DataFrame, cols: list = None,
                          hue: str = None, n_sample: int = 500,
                          save_path: str = None):
    df_s = df.sample(min(n_sample, len(df)), random_state=42)
    if cols is None:
        cols = ["Temperature", "Rainfall", "N", "P", "K", "Yield"]
    cols = [c for c in cols if c in df_s.columns]

    g = sns.pairplot(df_s[cols + ([hue] if hue and hue in df_s.columns else [])],
                     hue=hue if hue and hue in df_s.columns else None,
                     plot_kws={"alpha": 0.4, "s": 15},
                     diag_kind="kde", corner=True)
    g.figure.suptitle("Feature Pair Plot", y=1.01, fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        g.figure.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# 7. Partial Dependence Plot (simple 1-D)
# ─────────────────────────────────────────────────────────────────────────────

def plot_partial_dependence(model, X: pd.DataFrame, feature: str,
                             n_points: int = 50, save_path: str = None):
    """Simple 1-D partial dependence for a given feature."""
    if feature not in X.columns:
        print(f"[PDP] Column '{feature}' not in data.")
        return

    X_copy  = X.copy()
    grid    = np.linspace(X[feature].min(), X[feature].max(), n_points)
    preds   = []
    for val in grid:
        X_tmp = X_copy.copy()
        X_tmp[feature] = val
        preds.append(model.predict(X_tmp).mean())

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(grid, preds, color="#E15759", lw=2)
    ax.fill_between(grid, preds, alpha=0.15, color="#E15759")
    ax.set_xlabel(feature)
    ax.set_ylabel("Partial Dependence (Mean Predicted Yield)")
    ax.set_title(f"Partial Dependence Plot — {feature}", fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
