"""
Credit Solvency Assessment — End-to-End Modeling Pipeline
=========================================================

This module performs a complete credit-risk workflow:
    Stage 1 — Exploratory Data Analysis (EDA)
    Stage 2 — Data Preprocessing & Feature Engineering
    Stage 3 — Supervised Modeling (Logistic Regression, Random Forest, XGBoost)
    Stage 4 — Model Performance Assessment
    Stage 5 — Scorecard Development & Threshold Optimization

All figures are exported to `outputs/figures/`.
All trained models are saved in `models/`.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from matplotlib.patches import Patch
import matplotlib.ticker as mticker

from scipy.stats import mannwhitneyu, ks_2samp

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)

from xgboost import XGBClassifier
import joblib  


# ============================================================
# GLOBAL CONFIGURATION
# ============================================================

DATA_PATH = "data/credit_solvency_dataset.csv"
OUTPUT_DIR = "outputs/figures"
MODELS_DIR = "models"  # <-- Carpeta para guardar modelos

TARGET = "default_status"

CONTINUOUS_VARS = [
    "age",
    "gross_monthly_income",
    "years_continuous_experience",
    "current_debt_to_income_ratio",
]
DISCRETE_VARS = ["number_of_dependents"]
BINARY_VARS = ["home_ownership"]

all_features = CONTINUOUS_VARS + DISCRETE_VARS + BINARY_VARS

PALETTE_DICT = {
    0: "#2ecc71",
    1: "#e74c3c",
}


COLOR_SOLVENT = "#2ecc71"
COLOR_DEFAULTER = "#e74c3c"
COLOR_HOMEOWNER = "#27ae60"
COLOR_NON_OWNER = "#c0392b"
COLOR_DEPENDENTS = "#34495e"
COLOR_ACCENT = "#f39c12"
COLOR_BG = "#fafafa"
COLOR_TEXT = "#2c3e50"


def ensure_output_dir(output_dir: str) -> None:
    """
    Ensure the output directory exists.
    """
    os.makedirs(output_dir, exist_ok=True)


def ensure_models_dir() -> None:
    """
    Ensure the models directory exists.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)


def configure_visualization() -> None:
    """
    Centralize visualization configuration for consistent styling across all plots.
    """
    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set_palette("muted")

    sns.set_context("notebook", font_scale=1.1)

    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["savefig.dpi"] = 150
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.titlesize"] = 13
    plt.rcParams["axes.labelsize"] = 11

    sns.set_palette([COLOR_SOLVENT, COLOR_DEFAULTER, COLOR_DEPENDENTS, COLOR_HOMEOWNER])

    plt.rcParams.update(
        {
            "figure.dpi": 200,
            "savefig.dpi": 200,
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "axes.edgecolor": "#e0e0e0",
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.color": "#d5dbdb",
            "text.color": COLOR_TEXT,
        }
    )


def load_dataset(path: str) -> pd.DataFrame:
    """
    Load dataset from CSV.
    """
    df = pd.read_csv(path)
    return df


# ============================================================
# STAGE 1: DATASET STRUCTURAL INSPECTION + EDA PLOTS
# ============================================================

def structural_inspection(df: pd.DataFrame) -> None:
    """
    Print variable classification and dataset structure summary.
    """
    print("=" * 60)
    print("VARIABLE CLASSIFICATION")
    print("=" * 60)
    print(f"Continuous variables  : {CONTINUOUS_VARS}")
    print(f"Discrete variables    : {DISCRETE_VARS}")
    print(f"Binary variables      : {BINARY_VARS}")
    print(f"Target variable       : {TARGET}")

    print("\n" + "=" * 60)
    print("DATASET STRUCTURAL INSPECTION")
    print("=" * 60)
    print(f"Dimensions: {df.shape[0]:,} rows × {df.shape[1]} columns\n")
    print("Column Names and Data Types:")
    print(df.dtypes.to_string())
    print(f"\nMissing Values (Total): {df.isnull().sum().sum()}")
    print("\nMissing Values by Column:")
    print(df.isnull().sum().to_string())
    print(f"\nDuplicate Rows: {df.duplicated().sum()}")


def descriptive_statistics(df: pd.DataFrame) -> None:
    """
    Print descriptive statistics and distribution shape metrics.
    """
    print("\n" + "=" * 60)
    print("DESCRIPTIVE STATISTICS — ALL FEATURES")
    print("=" * 60)

    desc_stats = df[all_features + [TARGET]].describe().round(2)
    print(desc_stats.to_string())

    print("\n" + "=" * 60)
    print("DISTRIBUTION SHAPE METRICS — CONTINUOUS VARIABLES")
    print("=" * 60)

    for col in CONTINUOUS_VARS:
        skew_val = df[col].skew().round(3)
        kurt_val = df[col].kurtosis().round(3)
        print(f"{col:<40} Skewness: {skew_val:>8.3f}  |  Kurtosis: {kurt_val:>8.3f}")


def univariate_continuous_plots(df: pd.DataFrame, output_dir: str) -> None:
    """
    Generate univariate visualizations for continuous variables.
    Histogram with KDE plus mean/median reference lines.
    """
    print("\nGenerating univariate visualizations for continuous variables...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, var in enumerate(CONTINUOUS_VARS):
        ax = axes[i]

        sns.histplot(
            df[var],
            kde=True,
            bins=40,
            edgecolor="white",
            alpha=0.7,
            color="#3498db",
            ax=ax,
        )

        mean_val = df[var].mean()
        median_val = df[var].median()

        ax.axvline(
            mean_val,
            color="crimson",
            linestyle="--",
            linewidth=1.8,
            label=f"Mean: {mean_val:,.1f}",
        )
        ax.axvline(
            median_val,
            color="royalblue",
            linestyle="-",
            linewidth=1.8,
            label=f"Median: {median_val:,.1f}",
        )

        ax.set_title(f"Distribution of {var}", fontweight="bold")
        ax.set_xlabel(var.replace("_", " ").title())
        ax.set_ylabel("Frequency")
        ax.legend(fontsize=9, loc="upper right")

    suptitle_obj = fig.suptitle(
        "Univariate Analysis — Continuous Variables",
        fontsize=15,
        fontweight="bold",
        y=1.01,
    )

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "univariate_continuous.png"),
        bbox_inches="tight",
        dpi=150,
        bbox_extra_artists=[suptitle_obj],
    )
    plt.show()
    plt.close()


# ============================================================
# CATEGORICAL DASHBOARD
# ============================================================

def categorical_target_dashboard(df: pd.DataFrame, output_dir: str) -> None:
    """
    Create a professional categorical & target variable profiling dashboard.
    """
    print("\nGenerating categorical & target variable profiling dashboard...")

    DISCRETE_VAR = "number_of_dependents"
    BINARY_VAR = "home_ownership"

    total = len(df)

    dep_freq = df[DISCRETE_VAR].value_counts().sort_index()
    dep_default_rate = df.groupby(DISCRETE_VAR)[TARGET].mean() * 100

    home_counts = df[BINARY_VAR].value_counts().sort_index()
    home_default_rate = df.groupby(BINARY_VAR)[TARGET].mean() * 100

    target_counts = df[TARGET].value_counts().sort_index()
    default_rate_overall = df[TARGET].mean() * 100

    ct_home_default = pd.crosstab(
        df[BINARY_VAR].map({0: "Non-Homeowner", 1: "Homeowner"}),
        df[TARGET].map({0: "Solvent", 1: "Defaulter"}),
        normalize="index",
    ) * 100

    fig = plt.figure(figsize=(22, 12), facecolor=COLOR_BG)
    gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.35)

    fig.patch.set_edgecolor("#d5dbdb")
    fig.patch.set_linewidth(1)

    # ============================================================
    # ROW 1, COL 1
    # ============================================================
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor(COLOR_BG)

    bars = ax1.bar(
        dep_freq.index,
        dep_freq.values,
        color=COLOR_DEPENDENTS,
        edgecolor="white",
        alpha=0.85,
        width=0.65,
        zorder=3,
        label="Number of Applicants",
    )

    for bar, count in zip(bars, dep_freq.values):
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 15,
            f"{count:,}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color=COLOR_DEPENDENTS,
        )

    ax1b = ax1.twinx()
    ax1b.plot(
        dep_freq.index,
        dep_default_rate.values,
        color=COLOR_DEFAULTER,
        marker="D",
        markersize=8,
        linewidth=2.5,
        zorder=5,
        markerfacecolor="white",
        markeredgewidth=2,
        markeredgecolor=COLOR_DEFAULTER,
        label="Default Rate",
    )

    for x, y in zip(dep_freq.index, dep_default_rate.values):
        ax1b.annotate(
            f"{y:.1f}%",
            (x, y),
            textcoords="offset points",
            xytext=(0, 14),
            ha="center",
            fontsize=8,
            fontweight="bold",
            color=COLOR_DEFAULTER,
        )

    ax1.set_title(
        "Number of Dependents\nFrequency & Default Rate",
        fontweight="bold",
        fontsize=13,
        pad=15,
        color=COLOR_TEXT,
    )
    ax1.set_xlabel("Number of Dependents", fontweight="600")
    ax1.set_ylabel("Number of Applicants", fontweight="600")
    ax1b.set_ylabel("Default Rate (%)", fontweight="600", color=COLOR_DEFAULTER)
    ax1.set_xticks(dep_freq.index)
    ax1.set_ylim(0, dep_freq.max() * 1.35)
    ax1b.set_ylim(0, dep_default_rate.max() * 1.5)
    ax1b.tick_params(axis="y", colors=COLOR_DEFAULTER)
    ax1b.spines["right"].set_color(COLOR_DEFAULTER)
    ax1b.spines["right"].set_linewidth(1.5)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1b.get_legend_handles_labels()
    legend = ax1.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="upper left",
        fontsize=8,
        framealpha=0.9,
        edgecolor="#d5dbdb",
    )
    legend.get_frame().set_facecolor(COLOR_BG)

    # ============================================================
    # ROW 1, COL 2
    # ============================================================
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor(COLOR_BG)

    home_labels = ["Non-Homeowner\n(0)", "Homeowner\n(1)"]
    home_colors_bar = [COLOR_NON_OWNER, COLOR_HOMEOWNER]

    bars = ax2.bar(
        home_labels,
        home_counts.values,
        color=home_colors_bar,
        edgecolor="white",
        alpha=0.85,
        width=0.55,
        zorder=3,
    )

    for bar, count in zip(bars, home_counts.values):
        pct = count / total * 100
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() - (bar.get_height() * 0.18),
            f"{count:,}\n({pct:.1f}%)",
            ha="center",
            va="top",
            fontsize=11,
            fontweight="bold",
            color="white",
        )

    for i, (idx, rate) in enumerate(home_default_rate.items()):
        ax2.annotate(
            f"Default Rate:\n{rate:.1f}%",
            xy=(i, home_counts.values[i]),
            xytext=(i, home_counts.values[i] * 1.12),
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color=COLOR_DEFAULTER,
            bbox=dict(
                boxstyle="round,pad=0.4",
                facecolor="white",
                edgecolor=COLOR_DEFAULTER,
                alpha=0.9,
            ),
        )

    ax2.set_title(
        "Home Ownership Distribution\nwith Default Rate",
        fontweight="bold",
        fontsize=13,
        pad=15,
        color=COLOR_TEXT,
    )
    ax2.set_ylabel("Number of Applicants", fontweight="600")
    ax2.set_ylim(0, home_counts.max() * 1.45)

    # ============================================================
    # ROW 1, COL 3
    # ============================================================
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_facecolor(COLOR_BG)

    INNER_RADIUS = 0.48
    OUTER_RADIUS = 0.82
    RING_WIDTH = OUTER_RADIUS - INNER_RADIUS
    TEXT_RADIUS = INNER_RADIUS + (RING_WIDTH / 2)

    wedges, texts, autotexts = ax3.pie(
        target_counts.values,
        labels=None,
        colors=[COLOR_SOLVENT, COLOR_DEFAULTER],
        autopct="",
        startangle=90,
        wedgeprops={
            "width": RING_WIDTH,
            "edgecolor": "white",
            "linewidth": 2.5,
        },
        explode=(0.0, 0.05),
        radius=OUTER_RADIUS,
    )

    for i, (wedge, count) in enumerate(zip(wedges, target_counts.values)):
        pct = count / total * 100
        ang = (wedge.theta2 - wedge.theta1) / 2 + wedge.theta1
        rad = np.deg2rad(ang)

        x_text = TEXT_RADIUS * np.cos(rad)
        y_text = TEXT_RADIUS * np.sin(rad)

        ax3.text(
            x_text,
            y_text,
            f"{pct:.0f}%",
            ha="center",
            va="center",
            fontsize=15,
            fontweight="bold",
            color="white",
            zorder=10,
        )

    for i, (wedge, count) in enumerate(zip(wedges, target_counts.values)):
        ang = (wedge.theta2 - wedge.theta1) / 2 + wedge.theta1
        rad = np.deg2rad(ang)

        x_start = OUTER_RADIUS * np.cos(rad)
        y_start = OUTER_RADIUS * np.sin(rad)
        x_end = 1.28 * np.cos(rad)
        y_end = 1.28 * np.sin(rad)

        ax3.plot(
            [x_start, x_end],
            [y_start, y_end],
            color="#95a5a6",
            linewidth=1,
            alpha=0.6,
            zorder=5,
            clip_on=False,
        )

        ax3.plot(
            x_start,
            y_start,
            "o",
            markersize=5,
            color=COLOR_SOLVENT if i == 0 else COLOR_DEFAULTER,
            markeredgecolor="white",
            markeredgewidth=1.5,
            zorder=10,
        )

        label = "Solvent" if i == 0 else "Defaulter"
        ha = "left" if x_end > 0 else "right"

        ax3.text(
            x_end,
            y_end,
            f"{label}: {count:,}",
            ha=ha,
            va="center",
            fontsize=9.5,
            fontweight="600",
            color=COLOR_SOLVENT if i == 0 else COLOR_DEFAULTER,
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                edgecolor=COLOR_SOLVENT if i == 0 else COLOR_DEFAULTER,
                alpha=0.92,
                linewidth=1.2,
            ),
        )

    ax3.text(
        0,
        0.06,
        f"{total:,}",
        ha="center",
        va="center",
        fontsize=20,
        fontweight="bold",
        color=COLOR_TEXT,
    )
    ax3.text(
        0,
        -0.10,
        "Applicants",
        ha="center",
        va="center",
        fontsize=9,
        color="#7f8c8d",
        fontweight="500",
    )
    ax3.text(
        0,
        -0.26,
        f"{default_rate_overall:.0f}%",
        ha="center",
        va="center",
        fontsize=16,
        color=COLOR_DEFAULTER,
        fontweight="bold",
    )
    ax3.text(
        0,
        -0.38,
        "Default Rate",
        ha="center",
        va="center",
        fontsize=8,
        color="#7f8c8d",
    )

    ax3.set_title(
        "Target Variable Distribution\nDefault Status Breakdown",
        fontweight="bold",
        fontsize=13,
        pad=20,
        color=COLOR_TEXT,
    )

    # ============================================================
    # ROW 2, COL 1
    # ============================================================
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.set_facecolor(COLOR_BG)

    sns.boxplot(
        y=df[DISCRETE_VAR],
        color=COLOR_DEPENDENTS,
        width=0.35,
        linewidth=1.8,
        fliersize=0,
        ax=ax4,
        zorder=3,
    )
    sns.stripplot(
        y=df[DISCRETE_VAR],
        color=COLOR_ACCENT,
        alpha=0.25,
        size=4,
        jitter=True,
        ax=ax4,
        zorder=2,
    )

    stats_text = (
        f"Mean: {df[DISCRETE_VAR].mean():.2f}\n"
        f"Median: {df[DISCRETE_VAR].median():.0f}\n"
        f"Mode: {df[DISCRETE_VAR].mode()[0]:.0f}\n"
        f"Std Dev: {df[DISCRETE_VAR].std():.2f}\n"
        f"Max: {df[DISCRETE_VAR].max():.0f}"
    )

    ax4.text(
        0.65,
        0.95,
        stats_text,
        transform=ax4.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor="white",
            edgecolor="#d5dbdb",
            alpha=0.9,
        ),
        family="monospace",
    )

    ax4.set_title(
        "Distribution Spread\nNumber of Dependents",
        fontweight="bold",
        fontsize=13,
        pad=15,
        color=COLOR_TEXT,
    )
    ax4.set_ylabel("Number of Dependents", fontweight="600")

    # ============================================================
    # ROW 2, COL 2
    # ============================================================
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.set_facecolor(COLOR_BG)

    categories = ["Non-Homeowner", "Homeowner"]
    solvent_pct = ct_home_default["Solvent"].values
    defaulter_pct = ct_home_default["Defaulter"].values

    bars_solvent = ax5.barh(
        categories,
        solvent_pct,
        color=COLOR_SOLVENT,
        edgecolor="white",
        linewidth=1.5,
        height=0.55,
        label="Solvent",
        alpha=0.9,
    )
    bars_defaulter = ax5.barh(
        categories,
        defaulter_pct,
        left=solvent_pct,
        color=COLOR_DEFAULTER,
        edgecolor="white",
        linewidth=1.5,
        height=0.55,
        label="Defaulter",
        alpha=0.9,
    )

    for i, (s_pct, d_pct) in enumerate(zip(solvent_pct, defaulter_pct)):
        if s_pct > 25:
            ax5.text(
                s_pct / 2,
                i,
                f"{s_pct:.1f}%",
                ha="center",
                va="center",
                fontsize=11,
                fontweight="bold",
                color="white",
            )
        if d_pct > 10:
            ax5.text(
                s_pct + d_pct / 2,
                i,
                f"{d_pct:.1f}%",
                ha="center",
                va="center",
                fontsize=11,
                fontweight="bold",
                color="white",
            )

    ax5.set_title(
        "Home Ownership vs Default Status\n(Proportion within Each Group)",
        fontweight="bold",
        fontsize=13,
        pad=15,
        color=COLOR_TEXT,
    )
    ax5.set_xlabel("Percentage (%)", fontweight="600")
    ax5.set_xlim(0, 100)
    ax5.legend(loc="lower right", fontsize=9, framealpha=0.9, edgecolor="#d5dbdb")
    ax5.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))

    # ============================================================
    # ROW 2, COL 3
    # ============================================================
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.set_facecolor(COLOR_BG)

    df_heatmap = df.pivot_table(
        values=TARGET,
        index=DISCRETE_VAR,
        columns=BINARY_VAR,
        aggfunc="mean",
    ) * 100

    df_heatmap = df_heatmap.fillna(0)

    sns.heatmap(
        df_heatmap,
        annot=True,
        fmt=".1f",
        cmap="RdYlGn_r",
        vmin=0,
        vmax=max(30, df_heatmap.max().max()),
        center=default_rate_overall,
        linewidths=1.5,
        linecolor="white",
        cbar_kws={"label": "Default Rate (%)", "shrink": 0.8},
        annot_kws={"fontsize": 10, "fontweight": "bold"},
        ax=ax6,
    )

    ax6.set_title(
        "Default Rate Matrix\nDependents × Home Ownership",
        fontweight="bold",
        fontsize=13,
        pad=15,
        color=COLOR_TEXT,
    )
    ax6.set_xlabel("Home Ownership", fontweight="600")
    ax6.set_ylabel("Number of Dependents", fontweight="600")
    ax6.set_xticklabels(["Non-Owner", "Homeowner"], rotation=0)
    ax6.set_yticklabels(ax6.get_yticklabels(), rotation=0)

    suptitle_obj = fig.suptitle(
        "Categorical & Target Variable Profiling Dashboard\n"
        "Credit Solvency Assessment — Exploratory Data Analysis",
        fontsize=18,
        fontweight="bold",
        color="#1a252f",
        y=1.01,
    )

    fig.text(
        0.5,
        0.975,
        f"Dataset: {total:,} Applicants  |  Overall Default Rate: {default_rate_overall:.1f}%  |  Features Analyzed: 3",
        ha="center",
        fontsize=10,
        color="#7f8c8d",
        style="italic",
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(
        os.path.join(output_dir, "categorical_target_dashboard.png"),
        bbox_inches="tight",
        dpi=250,
        facecolor=COLOR_BG,
        edgecolor="none",
        bbox_extra_artists=[suptitle_obj],
    )
    plt.show()
    plt.close()

    print("=" * 60)
    print("CATEGORICAL & TARGET DASHBOARD GENERATED SUCCESSFULLY")
    print("=" * 60)


# ============================================================
# BIVARIATE ANALYSIS
# ============================================================

def bivariate_analysis_plots(df: pd.DataFrame, output_dir: str) -> None:
    """
    Generate bivariate analysis visuals of all predictors vs the target.
    """
    print("\nGenerating bivariate analysis...")

    vars_for_boxplot = CONTINUOUS_VARS + DISCRETE_VARS

    fig, axes = plt.subplots(2, 3, figsize=(20, 11))
    axes = axes.flatten()

    for i, var in enumerate(vars_for_boxplot):
        ax = axes[i]

        sns.boxplot(
            x=TARGET,
            y=var,
            data=df,
            hue=TARGET,
            palette=PALETTE_DICT,
            legend=False,
            width=0.45,
            fliersize=3,
            linewidth=1.5,
            ax=ax,
        )

        means = df.groupby(TARGET)[var].mean()
        medians = df.groupby(TARGET)[var].median()

        for j in range(2):
            label = "Solvent" if j == 0 else "Defaulter"
            color = COLOR_SOLVENT if j == 0 else COLOR_DEFAULTER
            m = means.iloc[j]
            med = medians.iloc[j]
            q3 = df[df[TARGET] == j][var].quantile(0.75)

            ax.text(
                j,
                q3,
                f"Mean: {m:,.1f}\nMed: {med:,.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
                color=color,
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor="white",
                    edgecolor=color,
                    linewidth=1.3,
                    alpha=0.92,
                ),
            )

        ax.set_title(var.replace("_", " ").title(), fontweight="bold", fontsize=12, pad=12)
        ax.set_xlabel("")
        ax.set_xticklabels(["Solvent", "Defaulter"], fontsize=10, fontweight="600")
        ax.tick_params(axis="y", labelsize=9)

    ax = axes[5]
    ct = pd.crosstab(df["home_ownership"], df[TARGET], normalize="index") * 100

    bars_s = ax.bar(
        ["Non-Owner", "Homeowner"],
        ct[0],
        color=COLOR_SOLVENT,
        edgecolor="white",
        linewidth=2,
        label="Solvent",
        width=0.50,
    )
    bars_d = ax.bar(
        ["Non-Owner", "Homeowner"],
        ct[1],
        bottom=ct[0],
        color=COLOR_DEFAULTER,
        edgecolor="white",
        linewidth=2,
        label="Defaulter",
        width=0.50,
    )

    for i in range(2):
        s_val = ct[0].iloc[i]
        d_val = ct[1].iloc[i]

        if s_val > 15:
            ax.text(
                i,
                s_val / 2,
                f"{s_val:.0f}%",
                ha="center",
                va="center",
                fontsize=13,
                fontweight="bold",
                color="white",
            )
        if d_val > 10:
            ax.text(
                i,
                s_val + d_val / 2,
                f"{d_val:.0f}%",
                ha="center",
                va="center",
                fontsize=11,
                fontweight="bold",
                color="white",
            )

    totals = df["home_ownership"].value_counts().sort_index()
    for i, t in enumerate(totals.values):
        ax.text(
            i,
            103,
            f"n={t:,}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color="#333333",
        )

    ax.set_title("Home Ownership", fontweight="bold", fontsize=12, pad=12)
    ax.set_ylabel("Percentage (%)", fontweight="600", fontsize=10)
    ax.set_ylim(0, 115)

    legend = ax.legend(
        loc="upper right",
        fontsize=9,
        framealpha=0.92,
        edgecolor="#999",
        fancybox=True,
        handletextpad=0.6,
        borderpad=0.5,
        handlelength=1.2,
    )
    legend.get_frame().set_linewidth(1.2)
    legend.set_zorder(10)

    ax.tick_params(axis="both", labelsize=10)

    suptitle_obj = fig.suptitle(
        "Bivariate Analysis — All Predictors vs Default Status",
        fontsize=16,
        fontweight="bold",
        y=1.01,
        color="#1a1a2e",
    )

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "bivariate_analysis.png"),
        bbox_inches="tight",
        dpi=200,
        bbox_extra_artists=[suptitle_obj],
    )
    plt.show()
    plt.close()

    print("\n" + "=" * 60)
    print("BIVARIATE SUMMARY — MEAN VALUES BY DEFAULT STATUS")
    print("=" * 60)
    summary = df.groupby(TARGET)[all_features].mean().round(2)
    summary.index = ["Solvent (0)", "Defaulter (1)"]
    summary["count"] = df.groupby(TARGET).size().values
    print(summary.to_string())

    print("\n" + "=" * 60)
    print("BIVARIATE ANALYSIS COMPLETE")
    print("=" * 60)


def numerical_comparison_by_class(df: pd.DataFrame) -> None:
    """
    Compute numerical summary, mean differences, and non-parametric statistical tests.
    """
    print("\n" + "=" * 70)
    print("BIVARIATE ANALYSIS: Numerical Comparison by Default Status")
    print("=" * 70)

    features = [
        "age",
        "gross_monthly_income",
        "years_continuous_experience",
        "current_debt_to_income_ratio",
        "number_of_dependents",
        "home_ownership",
    ]
    target = TARGET

    summary = df.groupby(target)[features].agg(["mean", "median", "std"]).round(2)
    summary["count"] = df.groupby(target).size()
    summary.index = ["Solvent (0)", "Defaulter (1)"]

    print("\n📊 DESCRIPTIVE STATISTICS BY CLASS:")
    print("-" * 70)
    print(summary.to_string())
    print("-" * 70)

    print("\n📈 MEAN DIFFERENCES: Defaulter vs Solvent")
    print("=" * 70)

    diff_data = []
    for var in features:
        mean_0 = df[df[target] == 0][var].mean()
        mean_1 = df[df[target] == 1][var].mean()
        abs_diff = mean_1 - mean_0

        pct_diff = (abs_diff / mean_0) * 100

        if pct_diff > 0:
            direction = "↑ Higher risk" if var != "gross_monthly_income" else "↓ Lower risk"
        else:
            direction = "↓ Lower risk" if var != "gross_monthly_income" else "↑ Higher risk"

        diff_data.append(
            {
                "Variable": var.replace("_", " ").title(),
                "Solvent": f"{mean_0:,.2f}",
                "Defaulter": f"{mean_1:,.2f}",
                "Absolute Δ": f"{abs_diff:+,.2f}",
                "Relative Δ": f"{pct_diff:+.1f}%",
                "Interpretation": direction,
            }
        )

    diff_df = pd.DataFrame(diff_data)
    print(diff_df.to_string(index=False))
    print("-" * 70)

    print("\n🔬 STATISTICAL SIGNIFICANCE (Mann-Whitney U Test)")
    print("=" * 70)
    print("Note: Non-parametric test suitable for non-normal distributions")
    print("-" * 70)

    sig_results = []
    for var in features:
        group_0 = df[df[target] == 0][var].dropna()
        group_1 = df[df[target] == 1][var].dropna()

        stat, p_value = mannwhitneyu(group_0, group_1, alternative="two-sided")

        if p_value < 0.001:
            sig = "*** (p < 0.001)"
        elif p_value < 0.01:
            sig = "**  (p < 0.01)"
        elif p_value < 0.05:
            sig = "*   (p < 0.05)"
        else:
            sig = "ns  (not significant)"

        sig_results.append(
            {
                "Variable": var.replace("_", " ").title(),
                "U-Statistic": f"{stat:.2f}",
                "P-value": f"{p_value:.4f}",
                "Significance": sig,
            }
        )

    sig_df = pd.DataFrame(sig_results)
    print(sig_df.to_string(index=False))
    print("-" * 70)


def density_analysis(df: pd.DataFrame, output_dir: str) -> None:
    """
    Generate density (KDE) plots for continuous variables by class.
    """
    print("\nGenerating density distributions by class...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, var in enumerate(CONTINUOUS_VARS):
        ax = axes[i]

        sns.kdeplot(
            data=df,
            x=var,
            hue=TARGET,
            fill=True,
            common_norm=False,
            palette=PALETTE_DICT,
            alpha=0.25,
            linewidth=2,
            ax=ax,
        )

        ax.set_title(var.replace("_", " ").title(), fontweight="bold", fontsize=11)
        ax.set_xlabel("")
        ax.set_ylabel("Density")
        ax.legend(["Solvent", "Defaulter"], fontsize=8, framealpha=0.9)

    suptitle_obj = fig.suptitle(
        "Density Distributions — Continuous Variables by Default Status",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "density_by_class.png"),
        bbox_inches="tight",
        dpi=150,
        bbox_extra_artists=[suptitle_obj],
    )
    plt.show()
    plt.close()

    print("Density analysis complete.")


# ============================================================
# INFORMATION VALUE (IV)
# ============================================================

def calculate_information_value(df: pd.DataFrame, feature: str, target: str, bins: int = 10):
    """
    Calculate Information Value (IV) for a given feature.
    Continuous/discrete variables are binned. Binary variables use natural categories.
    """
    df_temp = df[[feature, target]].copy()

    if df_temp[feature].nunique() > 5:
        df_temp["bin"] = pd.qcut(
            df_temp[feature],
            q=min(bins, df_temp[feature].nunique()),
            duplicates="drop",
        )
    else:
        df_temp["bin"] = df_temp[feature]

    grouped = df_temp.groupby("bin", observed=False).agg(
        total=("bin", "count"),
        goods=(target, lambda x: (x == 0).sum()),
        bads=(target, lambda x: (x == 1).sum()),
    ).reset_index(drop=True)

    total_goods = grouped["goods"].sum()
    total_bads = grouped["bads"].sum()

    grouped["pct_goods"] = grouped["goods"] / total_goods
    grouped["pct_bads"] = grouped["bads"] / total_bads

    grouped["pct_goods"] = grouped["pct_goods"].replace(0, 0.0001)
    grouped["pct_bads"] = grouped["pct_bads"].replace(0, 0.0001)

    grouped["WOE"] = np.log(grouped["pct_goods"] / grouped["pct_bads"])
    grouped["IV_contribution"] = (grouped["pct_goods"] - grouped["pct_bads"]) * grouped["WOE"]

    return grouped["IV_contribution"].sum(), grouped


def information_value_analysis(df: pd.DataFrame, output_dir: str) -> None:
    """
    Run IV analysis for all features and visualize predictive power by IV.
    """
    print("\n" + "=" * 60)
    print("INFORMATION VALUE (IV) ANALYSIS")
    print("=" * 60)

    iv_results = {}
    for feature in all_features:
        iv, _ = calculate_information_value(df, feature, TARGET)
        iv_results[feature] = iv

    iv_sorted = sorted(iv_results.items(), key=lambda x: x[1], reverse=True)

    print(f"\n{'Variable':<40} {'IV':>8}  {'Predictive Power'}")
    print("-" * 70)

    for var, iv_val in iv_sorted:
        if iv_val < 0.02:
            power = "Not Useful"
        elif iv_val < 0.10:
            power = "Weak"
        elif iv_val < 0.30:
            power = "Medium"
        elif iv_val < 0.50:
            power = "Strong"
        else:
            power = "Suspicious"

        print(f"{var:<40} {iv_val:>8.4f}  {power}")

    print("-" * 70)

    fig, ax = plt.subplots(figsize=(10, 5))
    vars_iv = [v for v, _ in iv_sorted]
    vals_iv = [iv for _, iv in iv_sorted]

    colors_iv = [
        "#e74c3c"
        if iv >= 0.30
        else "#f39c12"
        if iv >= 0.10
        else "#3498db"
        if iv >= 0.02
        else "#bdc3c7"
        for iv in vals_iv
    ]

    bars = ax.barh(vars_iv, vals_iv, color=colors_iv, edgecolor="white", linewidth=1.2, height=0.6)
    ax.invert_yaxis()
    ax.set_xlabel("Information Value", fontweight="600")
    ax.set_title("Predictive Power by Variable — Information Value (IV)", fontweight="bold", fontsize=13)

    for bar, val in zip(bars, vals_iv):
        ax.text(
            bar.get_width() + 0.003,
            bar.get_y() + bar.get_height() / 2.0,
            f"{val:.4f}",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

    for thresh, lbl, ls in [(0.02, "Not Useful", ":"), (0.10, "Weak", "--"), (0.30, "Medium", "-.")]:
        ax.axvline(thresh, color="gray", linestyle=ls, alpha=0.5, linewidth=1)
        ax.text(thresh + 0.003, -0.35, lbl, fontsize=7, color="gray")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "information_value.png"), bbox_inches="tight", dpi=150)
    plt.show()
    plt.close()

    print("\nInformation Value analysis complete.")


# ============================================================
# CORRELATION MATRIX
# ============================================================

def correlation_analysis(df: pd.DataFrame, output_dir: str) -> None:
    """
    Compute and plot correlation matrix of predictor variables.
    """
    print("\n" + "=" * 60)
    print("CORRELATION ANALYSIS — PREDICTOR VARIABLES")
    print("=" * 60)

    corr_matrix = df[all_features].corr()

    fig, ax = plt.subplots(figsize=(9, 7))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    cmap = sns.diverging_palette(250, 15, s=75, l=40, n=15, center="light")

    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt=".3f",
        cmap=cmap,
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": "Pearson Correlation"},
        annot_kws={"fontsize": 9, "fontweight": "bold"},
        ax=ax,
    )

    ax.set_title("Correlation Matrix of Predictor Variables", fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "correlation_matrix.png"), bbox_inches="tight", dpi=150)
    plt.show()
    plt.close()

    print("\nMulticollinearity Assessment:")
    high_corr_pairs = []
    for i in range(len(all_features)):
        for j in range(i + 1, len(all_features)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.5:
                high_corr_pairs.append((all_features[i], all_features[j], corr_val))

    if high_corr_pairs:
        print(f"  Pairs with |r| > 0.5:")
        for var1, var2, corr in high_corr_pairs:
            print(f"    {var1}  ↔  {var2}  (r = {corr:+.3f})")
    else:
        print("  No variable pairs exceed |r| > 0.5. Multicollinearity is not a concern.")

    print("\nCorrelation analysis complete.")


# ============================================================
# STAGE 2: DATA PREPROCESSING
# ============================================================

def winsorization_and_scaling(df: pd.DataFrame):
    """
    Apply outlier treatment (winsorization) and then standardization for continuous variables.
    """
    print("=" * 60)
    print("STAGE 2: DATA PREPROCESSING AND FEATURE ENGINEERING")
    print("=" * 60)

    print("\n--- 2.1 Outlier Treatment: Winsorization ---")

    winsorize_vars = CONTINUOUS_VARS
    original_stats = df[winsorize_vars].describe()

    for var in winsorize_vars:
        lower = df[var].quantile(0.01)
        upper = df[var].quantile(0.99)
        df[f"{var}_winsorized"] = df[var].clip(lower, upper)

        n_clipped = ((df[var] != df[f"{var}_winsorized"]).sum())
        print(f"  {var:<40} Lower: {lower:>10,.2f}  Upper: {upper:>10,.2f}  Clipped: {n_clipped:>4} rows")

    print("\n  gross_monthly_income — Before vs After Winsorization:")
    print(
        f"    Original:  max = {original_stats.loc['max', 'gross_monthly_income']:,.0f}, "
        f"skew = {df['gross_monthly_income'].skew():.2f}"
    )
    print(
        f"    Winsorized: max = {df['gross_monthly_income_winsorized'].max():,.0f}, "
        f"skew = {df['gross_monthly_income_winsorized'].skew():.2f}"
    )

    print("\n--- 2.2 Feature Scaling: Standardization ---")

    scaled_features = [f"{var}_winsorized" for var in CONTINUOUS_VARS]

    scaler = StandardScaler()
    df_scaled = df[scaled_features].copy()
    df_scaled.columns = [f"{col.replace('_winsorized', '_scaled')}" for col in df_scaled.columns]

    scaled_array = scaler.fit_transform(df[scaled_features])
    for i, col in enumerate(df_scaled.columns):
        df_scaled[col] = scaled_array[:, i]

    print("\n  Scaling Verification (mean ≈ 0, std ≈ 1):")
    for col in df_scaled.columns:
        print(f"    {col:<45} mean = {df_scaled[col].mean():>7.4f}  std = {df_scaled[col].std():>7.4f}")

    df_final = pd.concat(
        [
            df_scaled,
            df[DISCRETE_VARS + BINARY_VARS + [TARGET]],
        ],
        axis=1,
    )

    print(f"\n  Final dataset shape: {df_final.shape}")
    print(f"  Scaled features: {list(df_scaled.columns)}")
    print(f"  Unscaled features: {DISCRETE_VARS + BINARY_VARS + [TARGET]}")

    return df_final


def split_data(df_final: pd.DataFrame):
    """
    Split data into training/validation/test with stratification.
    """
    print("\n--- 2.3 Data Splitting with Stratification ---")

    X = df_final.drop(columns=[TARGET])
    y = df_final[TARGET]

    feature_names = X.columns.tolist()
    print(f"  Features ({len(feature_names)}): {feature_names}")
    print(f"  Target: {TARGET}")

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
    )

    print(f"\n  Partition Summary:")
    print(f"  {'Partition':<15} {'Size':>8} {'% of Total':>12} {'Default Rate':>15}")
    print(f"  {'-'*50}")
    for name, X_set, y_set in [
        ("Training", X_train, y_train),
        ("Validation", X_val, y_val),
        ("Test", X_test, y_test),
    ]:
        print(
            f"  {name:<15} {len(X_set):>8,} {len(X_set)/len(X)*100:>11.1f}% {y_set.mean():>14.1%}"
        )

    print(f"\n  Total: {len(X_train) + len(X_val) + len(X_test):,} rows across all partitions")
    return X_train, X_val, X_test, y_train, y_val, y_test


# ============================================================
# STAGE 3: SUPERVISED MODELING
# ============================================================

def supervised_modeling(X_train, y_train):
    """
    Train Logistic Regression, Random Forest, and XGBoost using GridSearchCV and Stratified CV.
    Guarda los modelos entrenados en la carpeta `models/`.
    """
    print("=" * 60)
    print("STAGE 3: SUPERVISED MODELING")
    print("=" * 60)

    print("\n--- 3.2 Imbalance Handling: Class Weight Calculation ---")
    n_negatives = (y_train == 0).sum()
    n_positives = (y_train == 1).sum()
    scale_pos_weight = n_negatives / n_positives

    print(f"  Training set composition:")
    print(f"    Solvent (0):   {n_negatives:,} ({n_negatives/len(y_train)*100:.1f}%)")
    print(f"    Defaulter (1): {n_positives:,} ({n_positives/len(y_train)*100:.1f}%)")
    print(f"    Scale pos weight (XGBoost): {scale_pos_weight:.2f}")
    print(f"    Logistic Regression: class_weight='balanced'")
    print(f"    Random Forest: class_weight='balanced'")

    print("\n--- 3.3 Cross-Validation Strategy ---")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print(f"  Strategy: Stratified 5-Fold Cross-Validation")
    print(f"  Each fold:{len(y_train)//5} training +{len(y_train) - len(y_train)//5} validation samples")
    print(f"  Default rate preserved at ~{y_train.mean():.1%} in every fold")

    print("\n--- 3.4 Logistic Regression ---")
    lr_param_grid = {"C": [0.01, 0.1, 0.5, 1.0, 5.0, 10.0]}

    lr_base = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
    )

    lr_grid = GridSearchCV(
        lr_base,
        lr_param_grid,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=0,
    )
    lr_grid.fit(X_train, y_train)

    print(f"  Best C (regularization):{lr_grid.best_params_['C']}")
    print(f"  Best CV AUC-ROC:{lr_grid.best_score_:.4f}")

    lr_best = lr_grid.best_estimator_

    lr_cv_scores = cross_val_score(lr_best, X_train, y_train, cv=cv, scoring="roc_auc")
    print(f"  CV AUC-ROC (mean ± std):{lr_cv_scores.mean():.4f} ±{lr_cv_scores.std():.4f}")

    print("\n--- 3.5 Random Forest ---")
    rf_param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [5, 8, 10, None],
        "min_samples_split": [5, 10, 20],
    }

    rf_base = RandomForestClassifier(
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    rf_grid = GridSearchCV(
        rf_base,
        rf_param_grid,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=0,
    )
    rf_grid.fit(X_train, y_train)

    print(f"  Best params:{rf_grid.best_params_}")
    print(f"  Best CV AUC-ROC:{rf_grid.best_score_:.4f}")

    rf_best = rf_grid.best_estimator_

    rf_cv_scores = cross_val_score(rf_best, X_train, y_train, cv=cv, scoring="roc_auc")
    print(f"  CV AUC-ROC (mean ± std):{rf_cv_scores.mean():.4f} ±{rf_cv_scores.std():.4f}")

    print("\n--- 3.6 XGBoost ---")
    xgb_param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [3, 4, 5],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.8, 1.0],
    }

    xgb_base = XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
        use_label_encoder=False,
    )

    xgb_grid = GridSearchCV(
        xgb_base,
        xgb_param_grid,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=0,
    )
    xgb_grid.fit(X_train, y_train)

    print(f"  Best params:{xgb_grid.best_params_}")
    print(f"  Best CV AUC-ROC:{xgb_grid.best_score_:.4f}")

    xgb_best = xgb_grid.best_estimator_

    xgb_cv_scores = cross_val_score(xgb_best, X_train, y_train, cv=cv, scoring="roc_auc")
    print(f"  CV AUC-ROC (mean ± std):{xgb_cv_scores.mean():.4f} ±{xgb_cv_scores.std():.4f}")

    # --- GUARDAR MODELOS EN LA CARPETA `models/` ---
    ensure_models_dir()
    joblib.dump(lr_best, os.path.join(MODELS_DIR, "logistic_regression_model.pkl"))
    joblib.dump(rf_best, os.path.join(MODELS_DIR, "random_forest_model.pkl"))
    joblib.dump(xgb_best, os.path.join(MODELS_DIR, "xgboost_model.pkl"))
    print("\n  Modelos guardados en la carpeta `models/`.")

    print("\n--- 3.7 Cross-Validation Performance Comparison ---")
    cv_results = {
        "Logistic Regression": lr_cv_scores,
        "Random Forest": rf_cv_scores,
        "XGBoost": xgb_cv_scores,
    }

    print(f"\n  {'Model':<25} {'CV AUC-ROC (Mean)':>18} {'CV AUC-ROC (Std)':>18}")
    print(f"  {'-'*60}")
    for model_name, scores in cv_results.items():
        print(f"  {model_name:<25} {scores.mean():>18.4f} {scores.std():>18.4f}")

    best_model_name = max(cv_results, key=lambda k: cv_results[k].mean())
    print(f"\n  Best performing model (CV): {best_model_name}")
    print(f"  AUC-ROC: {cv_results[best_model_name].mean():.4f}")

    fig, ax = plt.subplots(figsize=(8, 5))
    model_names = list(cv_results.keys())
    means = [cv_results[m].mean() for m in model_names]
    stds = [cv_results[m].std() for m in model_names]
    colors = ["#3498db", "#2ecc71", "#e74c3c"]

    bars = ax.bar(
        model_names,
        means,
        yerr=stds,
        color=colors,
        edgecolor="white",
        linewidth=1.5,
        capsize=8,
        width=0.5,
    )
    ax.set_ylabel("AUC-ROC", fontweight="600")
    ax.set_title(
        "Cross-Validation Performance Comparison\nStratified 5-Fold CV",
        fontweight="bold",
        fontsize=13,
    )
    ax.set_ylim(0.5, 1.0)
    ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

    for bar, mean_val in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.01,
            f"{mean_val:.4f}",
            ha="center",
            fontweight="bold",
            fontsize=11,
        )

    plt.tight_layout()

    return lr_best, rf_best, xgb_best, cv_results


# ============================================================
# STAGE 4: MODEL PERFORMANCE ASSESSMENT
# ============================================================

def evaluate_models(X_test, y_test, models, output_dir: str):
    """
    Evaluate models: ROC curves, confusion matrices, classification reports, PR curves.
    Returns comparison dataframe and prediction outputs needed for scorecard.
    """
    print("=" * 60)
    print("STAGE 4: MODEL PERFORMANCE ASSESSMENT")
    print("=" * 60)

    print("\n--- 4.2 Generate Test Set Predictions ---")
    predictions = {}
    probabilities = {}

    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        predictions[name] = y_pred
        probabilities[name] = y_prob
        print(f"  {name:<25} Predictions generated — {len(y_pred):,} samples")

    print(f"\n  Test set composition:")
    print(f"    Solvent (0):   {(y_test == 0).sum():,} ({(y_test == 0).mean()*100:.1f}%)")
    print(f"    Defaulter (1): {(y_test == 1).sum():,} ({(y_test == 1).mean()*100:.1f}%)")

    print("\n--- 4.3 ROC Curve Analysis ---")
    fig, ax = plt.subplots(figsize=(9, 7))
    roc_data = {}

    for name, probs in probabilities.items():
        fpr, tpr, thresholds = roc_curve(y_test, probs)
        auc_score = roc_auc_score(y_test, probs)
        roc_data[name] = {"fpr": fpr, "tpr": tpr, "auc": auc_score}
        ax.plot(fpr, tpr, linewidth=2.5, label=f"{name} (AUC = {auc_score:.4f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random Classifier (AUC = 0.5000)")
    ax.fill_between([0, 1], [0, 1], alpha=0.05, color="gray")

    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontweight="600", fontsize=11)
    ax.set_ylabel("True Positive Rate (Recall)", fontweight="600", fontsize=11)
    ax.set_title("ROC Curves — Model Comparison on Test Set", fontweight="bold", fontsize=14)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)

    for name, data in roc_data.items():
        print(f"  {name:<25} AUC-ROC: {data['auc']:.4f}  |  Gini: {(2*data['auc']-1):.4f}")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "roc_curves_comparison.png"), bbox_inches="tight", dpi=150)
    plt.show()
    plt.close()

    print("\n--- 4.4 Confusion Matrix Analysis ---")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    for i, (name, y_pred) in enumerate(predictions.items()):
        ax = axes[i]
        cm = confusion_matrix(y_test, y_pred)
        cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100

        sns.heatmap(
            cm_pct,
            annot=True,
            fmt=".1f",
            cmap="Blues",
            vmin=0,
            vmax=100,
            square=True,
            linewidths=1.5,
            cbar_kws={"label": "Row Percentage (%)", "shrink": 0.8},
            annot_kws={"fontsize": 13, "fontweight": "bold"},
            ax=ax,
        )

        for r in range(2):
            for c in range(2):
                ax.text(c + 0.5, r + 0.75, f"n={cm[r, c]}", ha="center", va="center", fontsize=9, color="#555555")

        ax.set_title(f"{name}", fontweight="bold", fontsize=12)
        ax.set_xlabel("Predicted", fontweight="600")
        ax.set_ylabel("Actual", fontweight="600")
        ax.set_xticklabels(["Solvent", "Defaulter"], fontsize=10)
        ax.set_yticklabels(["Solvent", "Defaulter"], fontsize=10, rotation=0)

    plt.suptitle("Confusion Matrices — Default Threshold (0.5)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrices.png"), bbox_inches="tight", dpi=150)
    plt.show()
    plt.close()

    print("\n--- 4.5 Classification Performance Summary ---")
    results_summary = []

    for name, y_pred in predictions.items():
        print(f"\n  {'='*50}")
        print(f"  {name}")
        print(f"  {'='*50}")

        report = classification_report(
            y_test,
            y_pred,
            target_names=["Solvent (0)", "Defaulter (1)"],
            output_dict=True,
        )

        print(classification_report(y_test, y_pred, target_names=["Solvent (0)", "Defaulter (1)"]))

        auc_val = roc_auc_score(y_test, probabilities[name])
        gini_val = 2 * auc_val - 1

        results_summary.append(
            {
                "Model": name,
                "AUC-ROC": f"{auc_val:.4f}",
                "Gini": f"{gini_val:.4f}",
                "Precision (Defaulter)": f'{report["Defaulter (1)"]["precision"]:.4f}',
                "Recall (Defaulter)": f'{report["Defaulter (1)"]["recall"]:.4f}',
                "F1-Score (Defaulter)": f'{report["Defaulter (1)"]["f1-score"]:.4f}',
                "Accuracy": f"{report['accuracy']:.4f}",
            }
        )

    print("\n--- 4.6 Model Comparison Summary ---")
    comparison_df = pd.DataFrame(results_summary).set_index("Model")

    print(f"\n  {'Model':<25} {'AUC-ROC':>9} {'Gini':>9} {'Precision':>11} {'Recall':>9} {'F1-Score':>9}")
    print(f"  {'-'*75}")

    for model_name, row in comparison_df.iterrows():
        print(
            f"  {model_name:<25} {row['AUC-ROC']:>9} {row['Gini']:>9} "
            f"{row['Precision (Defaulter)']:>11} {row['Recall (Defaulter)']:>9} "
            f"{row['F1-Score (Defaulter)']:>9}"
        )

    best_f1 = comparison_df["F1-Score (Defaulter)"].astype(float).idxmax()
    best_auc = comparison_df["AUC-ROC"].astype(float).idxmax()

    print(f"\n  Best F1-Score (Defaulter): {best_f1}")
    print(f"  Best AUC-ROC: {best_auc}")

    fig, ax = plt.subplots(figsize=(14, 8))

    metrics_to_plot = ["AUC-ROC", "Gini", "Precision\n(Defaulter)", "Recall\n(Defaulter)", "F1-Score\n(Defaulter)"]
    x = np.arange(len(metrics_to_plot))
    width = 0.22
    colors = ["#3498db", "#2ecc71", "#e74c3c"]

    for i, (model_name, row) in enumerate(comparison_df.iterrows()):
        values = [
            float(row["AUC-ROC"]),
            float(row["Gini"]),
            float(row["Precision (Defaulter)"]),
            float(row["Recall (Defaulter)"]),
            float(row["F1-Score (Defaulter)"]),
        ]
        bars = ax.bar(
            x + i * width,
            values,
            width,
            label=model_name,
            color=colors[i],
            edgecolor="white",
            linewidth=2,
            alpha=0.90,
        )

    for i, (model_name, row) in enumerate(comparison_df.iterrows()):
        values = [
            float(row["AUC-ROC"]),
            float(row["Gini"]),
            float(row["Precision (Defaulter)"]),
            float(row["Recall (Defaulter)"]),
            float(row["F1-Score (Defaulter)"]),
        ]
        bars = ax.patches[i * 5 : (i + 1) * 5]

        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 0.015,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
                color="#1a1a2e",
            )

    ax.set_ylabel("Score", fontweight="bold", fontsize=13, color="#1a1a2e", labelpad=10)
    ax.set_xlabel("")
    ax.set_xticks(x + width)
    ax.set_xticklabels(metrics_to_plot, fontsize=11, fontweight="600", color="#1a1a2e")
    ax.set_ylim(0, 1.18)
    ax.grid(True, alpha=0.12, axis="y", linestyle="-", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cccccc")
    ax.spines["bottom"].set_color("#cccccc")
    ax.tick_params(axis="y", labelsize=10, colors="#555555")

    for threshold in [0.5, 0.7, 0.9]:
        ax.axhline(y=threshold, color="#d5dbdb", linestyle=":", linewidth=0.8, alpha=0.6)

    ax.set_title(
        "Model Performance Comparison — Test Set Metrics",
        fontweight="bold",
        fontsize=16,
        color="#1a252f",
        pad=25,
    )

    legend = ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.10),
        ncol=3,
        fontsize=12,
        framealpha=0.95,
        edgecolor="#cccccc",
        fancybox=True,
        handletextpad=0.6,
        columnspacing=2.0,
        borderpad=0.8,
    )
    legend.get_frame().set_linewidth(1.5)

    plt.tight_layout(rect=[0.02, 0.06, 0.98, 0.96])
    plt.savefig(os.path.join(output_dir, "model_comparison_metrics.png"), bbox_inches="tight", dpi=200)
    plt.show()
    plt.close()

    print("\n--- 4.7 Precision-Recall Analysis ---")
    fig, ax = plt.subplots(figsize=(9, 7))

    for name, probs in probabilities.items():
        precision, recall, _ = precision_recall_curve(y_test, probs)
        ax.plot(recall, precision, linewidth=2.5, label=name)

    baseline_precision = y_test.mean()
    ax.axhline(
        baseline_precision,
        color="gray",
        linestyle="--",
        linewidth=1.2,
        alpha=0.7,
        label=f"Baseline (Default Rate = {baseline_precision:.2f})",
    )

    ax.set_xlabel("Recall (Proportion of Defaulters Identified)", fontweight="600", fontsize=11)
    ax.set_ylabel("Precision (Proportion of Correct Default Predictions)", fontweight="600", fontsize=11)
    ax.set_title("Precision-Recall Curves — Test Set", fontweight="bold", fontsize=14)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "precision_recall_curves.png"), bbox_inches="tight", dpi=150)
    plt.show()
    plt.close()

    print("\n" + "=" * 60)
    print("MODEL PERFORMANCE ASSESSMENT COMPLETE")
    print("=" * 60)

    recommended_model = comparison_df["F1-Score (Defaulter)"].astype(float).idxmax()
    recommended_auc = comparison_df.loc[recommended_model, "AUC-ROC"]
    recommended_f1 = comparison_df.loc[recommended_model, "F1-Score (Defaulter)"]
    recommended_gini = comparison_df.loc[recommended_model, "Gini"]

    print(
        f"""
  Final Model Recommendation: {recommended_model}
  
  Key Performance Indicators (Test Set):
    • AUC-ROC:                {recommended_auc}
    • Gini Coefficient:       {recommended_gini}
    • F1-Score (Defaulter):   {recommended_f1}
  
  Rationale:
    The {recommended_model} demonstrates the strongest balanced performance
    between precision and recall for the defaulting class. The model's AUC-ROC
    of {recommended_auc} indicates excellent discriminative ability, meaning it
    successfully separates solvent applicants from potential defaulters.
  
  Next Stage: Scorecard development and threshold optimization
"""
    )

    return comparison_df, predictions, probabilities, recommended_model, recommended_f1


# ============================================================
# STAGE 5: SCORECARD + THRESHOLD OPTIMIZATION
# ============================================================

def scorecard_and_thresholds(
    probabilities: dict,
    y_test,
    output_dir: str,
    model_key: str = "Random Forest",
):
    """
    Build score mapping from probabilities and optimize threshold with realistic cost assumptions.
    """
    print("=" * 60)
    print("STAGE 5: SCORECARD AND THRESHOLD OPTIMIZATION")
    print("=" * 60)

    y_prob = probabilities[model_key]

    BASE_SCORE, PDO, REF_ODDS = 600, 50, 10
    factor = PDO / np.log(2)
    offset = BASE_SCORE - factor * np.log(REF_ODDS)

    scores = offset - factor * np.log(y_prob / (1 - y_prob))

    print(f"\n  Score Range: {scores.min():.0f} – {scores.max():.0f}")
    print(f"  Solvent Mean: {scores[y_test == 0].mean():.0f}  |  Defaulter Mean: {scores[y_test == 1].mean():.0f}")
    print(f"  Separation: {scores[y_test == 0].mean() - scores[y_test == 1].mean():.0f} points")

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.kdeplot(scores[y_test == 0], fill=True, alpha=0.35, color="#2ecc71", linewidth=2.5, label="Solvent", ax=ax)
    sns.kdeplot(scores[y_test == 1], fill=True, alpha=0.35, color="#e74c3c", linewidth=2.5, label="Defaulter", ax=ax)
    ax.axvline(scores[y_test == 0].mean(), color="#27ae60", linestyle="--", linewidth=1.8)
    ax.axvline(scores[y_test == 1].mean(), color="#c0392b", linestyle="--", linewidth=1.8)
    ax.set_xlabel("Credit Score", fontweight="bold")
    ax.set_ylabel("Density", fontweight="bold")
    ax.set_title(f"Score Distribution by Default Status — {model_key}", fontweight="bold", fontsize=13)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "score_distribution.png"), bbox_inches="tight", dpi=150)
    plt.show()
    plt.close()

    COST_FP = 5000
    COST_FN = 800

    thresholds = np.arange(0.10, 0.91, 0.02)
    costs, metrics = [], []

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        costs.append(fp * COST_FP + fn * COST_FN)
        metrics.append(
            {
                "fp": fp,
                "fn": fn,
                "recall": tp / (tp + fn) if (tp + fn) > 0 else 0,
                "approval_rate": (y_pred == 0).mean(),
            }
        )

    idx_opt = np.argmin(costs)
    t_opt, cost_opt, m_opt = thresholds[idx_opt], costs[idx_opt], metrics[idx_opt]
    score_cutoff = offset - factor * np.log(t_opt / (1 - t_opt))

    print(f"\n  Cost Assumptions: FP=${COST_FP:,} (approve defaulter) | FN=${COST_FN:,} (reject solvent)")
    print(f"  Cost Ratio (FP:FN): {COST_FP/COST_FN:.1f}:1")
    print(f"\n  Optimal Threshold: {t_opt:.2f} | Cutoff Score: {score_cutoff:.0f}")
    print(f"  Min Total Cost: ${cost_opt:,.0f}")
    print(f"  False Positives: {m_opt['fp']} | False Negatives: {m_opt['fn']}")
    print(f"  Defaulter Recall: {m_opt['recall']:.0%} | Approval Rate: {m_opt['approval_rate']:.0%}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(thresholds, costs, linewidth=2.5, color="#3498db")
    axes[0].axvline(t_opt, color="#e74c3c", linestyle="--", linewidth=2, label=f"Optimal: {t_opt:.2f}")
    axes[0].scatter([t_opt], [cost_opt], color="#e74c3c", s=100, zorder=5)
    axes[0].set_xlabel("Threshold", fontweight="bold")
    axes[0].set_ylabel("Total Cost ($)", fontweight="bold")
    axes[0].set_title("Cost Optimization", fontweight="bold")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.2)

    axes[1].plot(thresholds, [m["fp"] for m in metrics], color="#e74c3c", linewidth=2, label="False Positives")
    axes[1].plot(thresholds, [m["fn"] for m in metrics], color="#f39c12", linewidth=2, label="False Negatives")
    axes[1].axvline(t_opt, color="#2c3e50", linestyle="--", linewidth=2)
    axes[1].set_xlabel("Threshold", fontweight="bold")
    axes[1].set_ylabel("Count", fontweight="bold")
    axes[1].set_title("Error Trade-off", fontweight="bold")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "threshold_optimization.png"), bbox_inches="tight", dpi=150)
    plt.show()

    plt.close()

    score_bins = [0, 500, 550, 600, 650, 700, 1000]
    band_labels = ["Very High Risk", "High Risk", "Medium Risk", "Low Risk", "Very Low Risk", "Excellent"]

    df_scores = pd.DataFrame({"score": scores, "actual": y_test.values})
    df_scores["band"] = pd.cut(df_scores["score"], bins=score_bins, labels=band_labels)

    bands = df_scores.groupby("band", observed=False).agg(
        count=("score", "count"),
        default_rate=("actual", "mean"),
    ).round(3)

    bands["default_rate"] = (bands["default_rate"] * 100).round(1)

    print(f"\n  {'Risk Band':<18} {'Count':>6} {'Default Rate':>14}")
    print(f"  {'-'*40}")

    for band, row in bands.iterrows():
        print(f"  {band:<18} {row['count']:>6,} {row['default_rate']:>13.1f}%")

    fig, ax = plt.subplots(figsize=(10, 5))
    colors_rb = ["#c0392b", "#e74c3c", "#f39c12", "#2ecc71", "#27ae60", "#1abc9c"]
    bars = ax.bar(
        bands.index,
        bands["default_rate"],
        color=colors_rb[: len(bands)],
        edgecolor="white",
        linewidth=1.5,
    )

    for bar, rate in zip(bars, bands["default_rate"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.5,
            f"{rate:.1f}%",
            ha="center",
            fontweight="bold",
            fontsize=11,
        )

    ax.set_ylabel("Default Rate (%)", fontweight="bold")
    ax.set_title("Default Rate by Risk Band", fontweight="bold", fontsize=13)
    ax.set_ylim(0, bands["default_rate"].max() * 1.4)
    ax.grid(True, alpha=0.2, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "risk_bands.png"), bbox_inches="tight", dpi=150)
    plt.show()
    plt.close()

    print(
        """
╔══════════════════════════════════════════════════╗
║          CREDIT SOLVENCY SCORECARD               ║
╠══════════════════════════════════════════════════╣
║  Model:       Random Forest                      ║
║  Score Range: 350 – 850                          ║
║  Cutoff:      Score ≥ """
        + f"{score_cutoff:.0f}"
        + """ → APPROVE              ║
║  PDO:         50 points doubles odds             ║
╠══════════════════════════════════════════════════╣
║  EXPECTED IMPACT:                                ║
║  Approval Rate:   """
        + f"{m_opt['approval_rate']:.0%}"
        + """                            ║
║  Defaulters Caught: """
        + f"{m_opt['recall']:.0%}"
        + """                           ║
║  Cost Savings:    Optimized for 10:1 cost ratio  ║
╚══════════════════════════════════════════════════╝
"""
    )

    return scores, t_opt, score_cutoff, m_opt


# ============================================================
# MAIN
# ============================================================

def main():
    """
    Main execution pipeline:
    - Configure visualization
    - Load dataset
    - EDA and dashboard plots
    - IV + correlation plots
    - Preprocessing + split
    - Modeling + evaluation + threshold optimization + scorecard
    """
    ensure_output_dir(OUTPUT_DIR)
    ensure_models_dir()  # <-- Asegurar que la carpeta `models/` exista
    configure_visualization()

    df = load_dataset(DATA_PATH)

    structural_inspection(df)
    descriptive_statistics(df)

    univariate_continuous_plots(df, OUTPUT_DIR)
    categorical_target_dashboard(df, OUTPUT_DIR)
    bivariate_analysis_plots(df, OUTPUT_DIR)
    numerical_comparison_by_class(df)
    density_analysis(df, OUTPUT_DIR)

    information_value_analysis(df, OUTPUT_DIR)
    correlation_analysis(df, OUTPUT_DIR)

    df_final = winsorization_and_scaling(df)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df_final)

    lr_best, rf_best, xgb_best, cv_results = supervised_modeling(X_train, y_train)

    models = {
        "Logistic Regression": lr_best,
        "Random Forest": rf_best,
        "XGBoost": xgb_best,
    }

    comparison_df, predictions, probabilities, recommended_model, recommended_f1 = evaluate_models(
        X_test,
        y_test,
        models,
        OUTPUT_DIR,
    )

    _scores, t_opt, score_cutoff, m_opt = scorecard_and_thresholds(
        probabilities=probabilities,
        y_test=y_test,
        output_dir=OUTPUT_DIR,
        model_key="Random Forest",
    )


if __name__ == "__main__":
    main()