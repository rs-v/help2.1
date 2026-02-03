"""Comprehensive VS analysis: ICE, validation, SHAP coupling, sensitivity, and uncertainty."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from xgboost import XGBRegressor
from sklearn.inspection import PartialDependenceDisplay, partial_dependence
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "dateset.csv"
VALIDATION_PATH = ROOT.parent / "数据验证数据" / "验证.xls"
OUTPUT_DIR = ROOT
FEATURES = [
    "S/L", "CD", "CV", "PW", "LP", "HP", "E", "v", "c", "φ", "T", "A", "n", "F",
]


def _clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if df.columns[-1] == "" or "Unnamed" in str(df.columns[-1]):
        df = df.drop(df.columns[-1], axis=1)
    return df


def _split_data(df: pd.DataFrame):
    X = df[FEATURES]
    y = df["VS"]
    return train_test_split(X, y, test_size=0.2, random_state=42)


def _fill_missing(train_df: pd.DataFrame, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    medians = train_df.median(numeric_only=True)
    return df.fillna(medians), medians


def _compute_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "R2": r2_score(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": mean_absolute_error(y_true, y_pred),
    }


def train_best_model():
    raw = pd.read_csv(DATA_PATH)
    df = _clean_dataset(raw)
    X_train, X_test, y_train, y_test = _split_data(df)
    X_train_filled, medians = _fill_missing(X_train, X_train)
    X_test_filled, _ = _fill_missing(X_train, X_test)

    model = XGBRegressor(
        n_estimators=100, 
        max_depth=6, 
        learning_rate=0.118, 
        random_state=42
    )

    model.fit(X_train_filled, y_train)

    train_pred = model.predict(X_train_filled)
    test_pred = model.predict(X_test_filled)
    metrics = {
        "train": _compute_metrics(y_train, train_pred),
        "test": _compute_metrics(y_test, test_pred),
    }
    return model, medians, (X_train_filled, X_test_filled, y_train, y_test, train_pred, test_pred, metrics)


def _set_tick_style(ax, size: int = 18, weight: str = "bold"):
    ax.tick_params(axis="both", labelsize=size)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight(weight)


def plot_ice_curves(model, X: pd.DataFrame, output_path: Path):
    breakpoints = []
    for feat in FEATURES:
        pd_feat = partial_dependence(model, X, [feat], grid_resolution=40)
        grid_values = pd_feat.get("values") or pd_feat.get("grid_values")
        if grid_values is None:
            breakpoints.append({"feature": feat, "x": np.nan, "y": np.nan, "max_abs_slope": np.nan})
            continue
        grid = np.ravel(grid_values[0])
        avg = np.ravel(pd_feat["average"][0])
        if len(grid) < 2:
            breakpoints.append({"feature": feat, "x": np.nan, "y": np.nan, "max_abs_slope": np.nan})
            continue
        slopes = np.gradient(avg, grid)
        change_idx = int(np.argmax(np.abs(slopes)))
        breakpoints.append(
            {
                "feature": feat,
                "x": float(grid[change_idx]),
                "y": float(avg[change_idx]),
                "max_abs_slope": float(slopes[change_idx]),
            }
        )

    chunk_size = 9
    image_paths = []
    for chunk_idx, start in enumerate(range(0, len(FEATURES), chunk_size)):
        feats = FEATURES[start : start + chunk_size]
        fig, axes = plt.subplots(3, 3, figsize=(20, 18), sharey=False)
        axes_flat = axes.flatten()
        target_axes = axes_flat[: len(feats)]
        display = PartialDependenceDisplay.from_estimator(
            model,
            X,
            feats,
            kind="both",
            grid_resolution=40,
            subsample=120,
            random_state=42,
            ice_lines_kw={"color": "#b0b0b0", "alpha": 0.35, "linewidth": 0.8},
            pd_line_kw={"color": "#d62728", "linewidth": 2.4, "label": "Average"},
            ax=target_axes,
        )

        for ax in axes_flat[len(feats):]:
            ax.set_visible(False)

        for ax_idx, (ax, feat) in enumerate(zip(target_axes, feats)):
            ax.set_xlabel(feat, fontsize=18, fontweight="bold")
            ax.set_ylabel("Predicted VS", fontsize=18, fontweight="bold")
            _set_tick_style(ax)
            ax.grid(False)
            lines = display.lines_[ax_idx]
            if len(lines) > 0:
                avg_line = lines[0]
                avg_line.set_label("Average")
                ax.legend(handles=[avg_line], fontsize=18, frameon=False, loc="best")

        fig.suptitle("ICE and PDP (GBM)", fontsize=22)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        chunk_path = output_path.with_name(f"{output_path.stem}_part{chunk_idx + 1}.png")
        fig.savefig(chunk_path, dpi=600, bbox_inches="tight")
        plt.close(fig)
        image_paths.append(chunk_path)

    breakpoints_path = output_path.with_name(f"{output_path.stem}_change_points.xlsx")
    pd.DataFrame(breakpoints).to_excel(breakpoints_path, index=False)
    return image_paths, breakpoints_path


def validate_on_new_data(model, medians: pd.Series, output_prefix: Path):
    val_df = _clean_dataset(pd.read_excel(VALIDATION_PATH))
    val_df = val_df.dropna(subset=["VS"]).reset_index(drop=True)
    val_features = val_df[FEATURES].fillna(medians)
    y_val = val_df["VS"]
    preds = model.predict(val_features)

    metrics = _compute_metrics(y_val, preds)
    metrics_df = pd.DataFrame([metrics])
    metrics_path = output_prefix.with_name(f"{output_prefix.name}_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_val, preds, alpha=0.7, color="#1f77b4", edgecolors="k", s=45, label="Predictions")
    lims = [min(y_val.min(), preds.min()), max(y_val.max(), preds.max())]
    ax.plot(lims, lims, "r--", linewidth=2, label="1:1 Line")
    ax.set_xlabel("Experimental VS")
    ax.set_ylabel("Predicted VS")
    ax.set_title("Validation: Experimental vs Predicted")
    ax.legend()
    ax.grid(True, alpha=0.3)
    text = f"R²={metrics['R2']:.3f}\nRMSE={metrics['RMSE']:.3f}\nMAE={metrics['MAE']:.3f}"
    ax.text(0.05, 0.95, text, transform=ax.transAxes, va="top", ha="left",
            bbox={"facecolor": "white", "alpha": 0.8, "pad": 6})
    fig.tight_layout()
    scatter_path = output_prefix.with_name(f"{output_prefix.name}_scatter.png")
    fig.savefig(scatter_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    return metrics_df, metrics_path, scatter_path


def shap_analysis(model, X: pd.DataFrame, output_prefix: Path):
    sample_size = min(400, len(X))
    sample = X.sample(sample_size, random_state=42)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)

    dep_dir = output_prefix.parent / "shap_dependence_plots"
    dep_dir.mkdir(parents=True, exist_ok=True)

    mean_abs = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({"feature": sample.columns, "mean_abs_shap": mean_abs})
    importance_df = importance_df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    importance_path = output_prefix.with_name(f"{output_prefix.name}_importance.csv")
    importance_df.to_csv(importance_path, index=False)

    # Bar plot with value labels
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.barplot(data=importance_df, x="mean_abs_shap", y="feature", ax=ax, color="#4c72b0")
    for i, val in enumerate(importance_df["mean_abs_shap"]):
        ax.text(val, i, f"{val:.3f}", va="center", ha="left", fontsize=18)
    ax.set_xlabel("mean(|SHAP|)", fontsize=18, fontweight="bold")
    ax.set_ylabel("Feature", fontsize=18, fontweight="bold")
    _set_tick_style(ax)
    ax.set_title("SHAP Feature Importance (global)")
    fig.tight_layout()
    bar_path = output_prefix.with_name(f"{output_prefix.name}_importance_bar.png")
    fig.savefig(bar_path, dpi=600, bbox_inches="tight")
    plt.close(fig)

    # Pie chart of SHAP contributions with bottom 5 grouped as other features
    labels = importance_df["feature"].tolist()
    values = importance_df["mean_abs_shap"].tolist()
    if len(labels) > 5:
        keep_n = len(labels) - 5
        pie_labels = labels[:keep_n]
        pie_values = values[:keep_n]
        pie_labels.append("other features")
        pie_values.append(sum(values[keep_n:]))
    else:
        pie_labels, pie_values = labels, values
    total_value = sum(pie_values) if pie_values else 0
    fig_pie, ax_pie = plt.subplots(figsize=(8, 8))
    wedges, texts, autotexts = ax_pie.pie(
        pie_values,
        labels=pie_labels,
        autopct=(lambda pct: f"{pct:.1f}% ({pct / 100 * total_value:.3g})"),
        startangle=90,
        textprops={"fontsize": 14},
    )
    for text in texts:
        text.set_fontsize(16)
        text.set_fontweight("bold")
    for autotext in autotexts:
        autotext.set_fontsize(14)
    ax_pie.set_title("SHAP share by feature", fontsize=18)
    fig_pie.tight_layout()
    pie_path = output_prefix.with_name(f"{output_prefix.name}_importance_pie.png")
    fig_pie.savefig(pie_path, dpi=600, bbox_inches="tight")
    plt.close(fig_pie)

    top5 = importance_df.head(5)["feature"].tolist()
    shap.summary_plot(shap_values, sample, plot_type="dot", feature_names=sample.columns,
                      max_display=5, show=False)
    plt.title("Top 5 Features: SHAP Summary")
    plt.tight_layout()
    summary_path = output_prefix.with_name(f"{output_prefix.name}_summary.png")
    plt.savefig(summary_path, dpi=600, bbox_inches="tight")
    plt.close()

    # Dependence plots for coupling effects
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes_flat = axes.flatten()
    for idx, feat in enumerate(top5):
        interaction = top5[(idx + 1) % len(top5)] if len(top5) > 1 else None
        shap.dependence_plot(feat, shap_values, sample, interaction_index=interaction,
                             show=False, ax=axes_flat[idx], dot_size=18)
        axes_flat[idx].set_title(f"{feat} vs SHAP (color: {interaction})")
        axes_flat[idx].grid(True, alpha=0.3)
    for ax in axes_flat[len(top5):]:
        ax.set_visible(False)
    fig.suptitle("SHAP Dependence (coupling effects)", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    dependence_path = output_prefix.with_name(f"{output_prefix.name}_dependence.png")
    fig.savefig(dependence_path, dpi=600, bbox_inches="tight")
    plt.close(fig)

    # Individual dependence plots for all features into dedicated folder
    dependence_files = []
    for feat in sample.columns:
        safe_name = feat.replace("/", "-").replace(" ", "_")
        fig_f, ax_f = plt.subplots(figsize=(6, 5))
        shap.dependence_plot(feat, shap_values, sample, interaction_index=None,
                             show=False, ax=ax_f, dot_size=20)
        ax_f.set_title(f"{feat} SHAP Dependence")
        ax_f.grid(True, alpha=0.3)
        fig_f.tight_layout()
        out_path = dep_dir / f"{output_prefix.name}_{safe_name}_dependence.png"
        fig_f.savefig(out_path, dpi=600, bbox_inches="tight")
        plt.close(fig_f)
        dependence_files.append(out_path)

    # SHAP value correlation heatmap across features
    shap_df = pd.DataFrame(shap_values, columns=sample.columns)
    corr = shap_df.corr().fillna(0)  # fill potential NaNs so the full matrix renders
    corr_sig = pd.DataFrame(
        np.vectorize(lambda v: float(f"{v:.4g}"))(corr.to_numpy()),
        index=corr.index,
        columns=corr.columns,
    )
    plt.figure(figsize=(10, 10))
    sns.heatmap(
        corr,
        cmap="RdBu_r",
        center=0,
        square=True,
        cbar_kws={"shrink": 0.8},
        linewidths=0.5,
        linecolor="white",
        annot=True,
        fmt=".4g",
        annot_kws={"size": 10},
    )
    plt.title("SHAP Value Correlation Heatmap (full matrix)")
    plt.tight_layout()
    heatmap_path = output_prefix.with_name(f"{output_prefix.name}_heatmap.png")
    plt.savefig(heatmap_path, dpi=600, bbox_inches="tight")
    plt.close()
    heatmap_csv_path = output_prefix.with_name(f"{output_prefix.name}_heatmap_values.csv")
    corr_sig.to_csv(heatmap_csv_path, float_format="%.4g")

    return (
        importance_df,
        shap_values,
        sample,
        {
            "importance_csv": importance_path,
            "importance_bar": bar_path,
            "importance_pie": pie_path,
            "summary_plot": summary_path,
            "dependence_plot": dependence_path,
            "dependence_dir": dep_dir,
            "dependence_files": dependence_files,
            "shap_heatmap": heatmap_path,
            "shap_heatmap_csv": heatmap_csv_path,
        },
    )


def main():
    model, medians, (X_train_filled, X_test_filled, y_train, y_test, train_pred, test_pred, metrics) = train_best_model()

    print("Train metrics:", metrics["train"])
    print("Test metrics:", metrics["test"])

    # Task 1: ICE grid
    ice_path = OUTPUT_DIR / "ice_curves.png"
    ice_images, change_points_path = plot_ice_curves(model, X_train_filled, ice_path)
    print(f"ICE grids saved to: {ice_images}")
    print(f"ICE change points saved to: {change_points_path}")

    # Task 2: Validation scatter and stats
    val_prefix = OUTPUT_DIR / "validation_results"
    val_metrics_df, val_metrics_path, val_scatter_path = validate_on_new_data(model, medians, val_prefix)
    print("Validation metrics:\n", val_metrics_df)
    print("Saved:", val_metrics_path, val_scatter_path)

    # Task 3: SHAP coupling analysis
    shap_prefix = OUTPUT_DIR / "shap_outputs"
    importance_df, shap_values, sample, shap_paths = shap_analysis(model, X_test_filled, shap_prefix)
    print("Top 5 SHAP features:\n", importance_df.head(5))
    print("SHAP artifacts:", shap_paths)


if __name__ == "__main__":
    main()
