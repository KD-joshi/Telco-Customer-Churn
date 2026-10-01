"""
Exploratory Data Analysis for the Telco Customer Churn dataset.

Generates all visual and statistical analyses, saves figures to EDA/figures/,
and prints a structured observation summary.

Usage:
    python EDA/eda.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving figures

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns

# ── Setup ──────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_PATH = PROJECT_ROOT / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Visual styling
sns.set_theme(style="whitegrid", font_scale=1.1)
PALETTE_CHURN = {"No": "#3498db", "Yes": "#e74c3c"}
DPI = 150


def save_fig(fig, name):
    """Save figure and close to free memory."""
    fig.savefig(FIG_DIR / name, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✓ Saved: figures/{name}")


# ── Load data ──────────────────────────────────────────────────────────────────

def load_raw():
    df = pd.read_csv(DATA_PATH)
    # Fix TotalCharges: stored as string, 11 rows have whitespace (tenure=0)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  1. DATASET OVERVIEW & DATA TYPES
# ══════════════════════════════════════════════════════════════════════════════

def dataset_overview(df):
    print("=" * 70)
    print("  1. DATASET OVERVIEW")
    print("=" * 70)
    print(f"\n  Rows   : {df.shape[0]}")
    print(f"  Columns: {df.shape[1]}")

    print(f"\n  Column types:")
    for col in df.columns:
        dtype = str(df[col].dtype)
        nunique = df[col].nunique()
        sample = df[col].dropna().iloc[0] if not df[col].dropna().empty else "N/A"
        print(f"    {col:20s}  dtype={dtype:10s}  unique={nunique:5d}  sample={sample}")

    print()


# ══════════════════════════════════════════════════════════════════════════════
#  2. MISSING VALUES ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def missing_values_analysis(df):
    print("=" * 70)
    print("  2. MISSING VALUES ANALYSIS")
    print("=" * 70)

    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({"count": missing, "percent": missing_pct})
    missing_df = missing_df[missing_df["count"] > 0]

    if missing_df.empty:
        print("\n  No null values detected at first glance.")
    else:
        print(f"\n  Columns with nulls:")
        print(missing_df.to_string())

    # TotalCharges deep-dive
    tc_null = df[df["TotalCharges"].isna()]
    print(f"\n  TotalCharges (parsed from string):")
    print(f"    - {len(tc_null)} rows have non-numeric values (empty strings in raw CSV)")
    print(f"    - All {len(tc_null)} correspond to tenure = 0 (brand-new signups)")
    print(f"    - Imputation strategy: fill with 0.0 (no charges accumulated yet)")
    print()

    # Missing values bar chart — far more readable than a heatmap when
    # only a single column has missing data.
    fig, ax = plt.subplots(figsize=(14, 5))
    missing_counts = df.isnull().sum()
    colors = ["#e74c3c" if v > 0 else "#95a5a6" for v in missing_counts.values]
    bars = ax.bar(missing_counts.index, missing_counts.values, color=colors,
                  edgecolor="white", linewidth=0.8)

    # Annotate the bars that have missing values
    for bar, val in zip(bars, missing_counts.values):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
                    f"{val}\n({val / len(df) * 100:.2f}%)",
                    ha="center", va="bottom", fontweight="bold", fontsize=11,
                    color="#e74c3c")

    ax.set_title("Missing Values per Column", fontsize=14, fontweight="bold")
    ax.set_ylabel("Missing Count")
    ax.tick_params(axis="x", rotation=45)
    ax.set_ylim(0, max(missing_counts.max() * 1.5, 20))  # ensure non-zero columns are visible
    ax.axhline(y=0, color="black", linewidth=0.8)
    save_fig(fig, "01_missing_values_barchart.png")

    # Detail view of TotalCharges nulls
    if len(tc_null) > 0:
        fig, ax = plt.subplots(figsize=(10, 4))
        detail_cols = ["customerID", "tenure", "MonthlyCharges", "TotalCharges", "Churn"]
        cell_text = tc_null[detail_cols].values.tolist()
        table = ax.table(cellText=cell_text, colLabels=detail_cols, loc="center",
                         cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        ax.axis("off")
        ax.set_title("Rows with Missing TotalCharges (all tenure = 0)",
                      fontsize=13, fontweight="bold", pad=20)
        save_fig(fig, "02_missing_totalcharges_detail.png")


# ══════════════════════════════════════════════════════════════════════════════
#  3. TARGET VARIABLE — CLASS IMBALANCE
# ══════════════════════════════════════════════════════════════════════════════

def target_distribution(df):
    print("=" * 70)
    print("  3. TARGET VARIABLE — CLASS IMBALANCE")
    print("=" * 70)

    counts = df["Churn"].value_counts()
    pcts = df["Churn"].value_counts(normalize=True) * 100

    print(f"\n  No (stayed)  : {counts['No']:5d}  ({pcts['No']:.1f}%)")
    print(f"  Yes (churned): {counts['Yes']:5d}  ({pcts['Yes']:.1f}%)")
    print(f"  Ratio        : {counts['No'] / counts['Yes']:.2f} : 1")
    print(f"\n  The dataset is moderately imbalanced — a naive 'always predict No'")
    print(f"  classifier would achieve {pcts['No']:.1f}% accuracy while missing every")
    print(f"  actual churner. This necessitates class-weighted loss functions")
    print(f"  and evaluation metrics beyond simple accuracy.")
    print()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Bar chart
    ax = axes[0]
    bars = ax.bar(counts.index, counts.values, color=[PALETTE_CHURN["No"], PALETTE_CHURN["Yes"]],
                  edgecolor="white", linewidth=1.5, width=0.5)
    for bar, count, pct in zip(bars, counts.values, pcts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                f"{count}\n({pct:.1f}%)", ha="center", va="bottom", fontweight="bold")
    ax.set_title("Churn Distribution", fontsize=13, fontweight="bold")
    ax.set_ylabel("Customer Count")
    ax.set_ylim(0, counts.max() * 1.2)

    # Pie chart
    ax = axes[1]
    wedges, texts, autotexts = ax.pie(
        counts.values, labels=counts.index, autopct="%1.1f%%",
        colors=[PALETTE_CHURN["No"], PALETTE_CHURN["Yes"]],
        startangle=90, explode=(0, 0.05), shadow=True,
        textprops={"fontsize": 12}
    )
    for t in autotexts:
        t.set_fontweight("bold")
    ax.set_title("Churn Proportion", fontsize=13, fontweight="bold")

    fig.suptitle("Target Variable Analysis: Class Imbalance", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_fig(fig, "03_target_distribution.png")


# ══════════════════════════════════════════════════════════════════════════════
#  4. NUMERIC FEATURE DISTRIBUTIONS
# ══════════════════════════════════════════════════════════════════════════════

def numeric_distributions(df):
    print("=" * 70)
    print("  4. NUMERIC FEATURE DISTRIBUTIONS")
    print("=" * 70)

    # Fill missing for plotting
    df_plot = df.copy()
    df_plot["TotalCharges"] = df_plot["TotalCharges"].fillna(0.0)

    numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]

    print(f"\n  Summary statistics:")
    print(df_plot[numeric_cols].describe().round(2).to_string())
    print()

    # Histograms + KDE split by Churn
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, col in zip(axes, numeric_cols):
        for label, color in PALETTE_CHURN.items():
            subset = df_plot[df_plot["Churn"] == label][col]
            ax.hist(subset, bins=30, alpha=0.5, label=f"Churn={label}",
                    color=color, edgecolor="white")
        ax.set_title(col, fontsize=13, fontweight="bold")
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
        ax.legend()

    fig.suptitle("Numeric Feature Distributions by Churn Status",
                 fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_fig(fig, "04_numeric_distributions.png")

    # Box plots by churn
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, col in zip(axes, numeric_cols):
        sns.boxplot(data=df_plot, x="Churn", y=col, hue="Churn",
                    palette=PALETTE_CHURN, ax=ax, width=0.4, legend=False)
        ax.set_title(col, fontsize=13, fontweight="bold")

    fig.suptitle("Numeric Features: Boxplots by Churn", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_fig(fig, "05_numeric_boxplots.png")

    # Observations
    print("  Key observations:")
    print("    - Tenure: churners cluster heavily at low tenure (0–10 months).")
    print("      Long-tenure customers are far more likely to stay.")
    print("    - MonthlyCharges: churners tend to have higher monthly charges.")
    print("    - TotalCharges: churners have lower totals, consistent with the")
    print("      low-tenure pattern (they leave before accumulating charges).")
    print()


# ══════════════════════════════════════════════════════════════════════════════
#  5. CATEGORICAL FEATURE DISTRIBUTIONS
# ══════════════════════════════════════════════════════════════════════════════

def categorical_distributions(df):
    print("=" * 70)
    print("  5. CATEGORICAL FEATURE DISTRIBUTIONS")
    print("=" * 70)

    cat_cols = [
        "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
        "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
        "Contract", "PaperlessBilling", "PaymentMethod",
    ]

    # Churn rate by each categorical feature
    fig, axes = plt.subplots(4, 4, figsize=(22, 20))
    axes = axes.flatten()

    for i, col in enumerate(cat_cols):
        ax = axes[i]
        ct = pd.crosstab(df[col], df["Churn"], normalize="index") * 100
        ct.plot(kind="bar", stacked=True, ax=ax,
                color=[PALETTE_CHURN["No"], PALETTE_CHURN["Yes"]],
                edgecolor="white", linewidth=0.5)
        ax.set_title(col, fontsize=11, fontweight="bold")
        ax.set_ylabel("Percentage")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=45)
        ax.legend(title="Churn", fontsize=8, title_fontsize=9)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())

    # Hide unused subplots
    for j in range(len(cat_cols), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Churn Rate by Categorical Feature",
                 fontsize=16, fontweight="bold", y=1.01)
    fig.tight_layout()
    save_fig(fig, "06_categorical_churn_rates.png")

    # Print notable patterns
    print("\n  Key observations:")
    print("    - Gender has virtually no effect on churn rate.")
    print("    - Senior citizens churn at nearly double the rate of non-seniors.")
    print("    - Month-to-month contracts have dramatically higher churn (~43%)")
    print("      vs. one-year (~11%) and two-year (~3%) contracts.")
    print("    - Fiber optic internet users churn at ~42%, far above DSL (~19%).")
    print("    - Customers without add-on services (OnlineSecurity, TechSupport,")
    print("      DeviceProtection) churn more — these services act as retention anchors.")
    print("    - Electronic check payment is associated with the highest churn rate.")
    print("    - Paperless billing customers churn more than non-paperless.")
    print()


# ══════════════════════════════════════════════════════════════════════════════
#  6. CORRELATION ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def correlation_analysis(df):
    print("=" * 70)
    print("  6. CORRELATION ANALYSIS")
    print("=" * 70)

    df_encoded = df.copy()
    df_encoded["TotalCharges"] = df_encoded["TotalCharges"].fillna(0.0)
    df_encoded["Churn_binary"] = (df_encoded["Churn"] == "Yes").astype(int)

    # Encode binary categoricals for correlation
    binary_map = {"Yes": 1, "No": 0, "Male": 1, "Female": 0}
    for col in ["gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling"]:
        df_encoded[col] = df_encoded[col].map(binary_map)

    numeric_for_corr = [
        "SeniorCitizen", "gender", "Partner", "Dependents",
        "tenure", "PhoneService", "PaperlessBilling",
        "MonthlyCharges", "TotalCharges", "Churn_binary",
    ]

    corr = df_encoded[numeric_for_corr].corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, vmin=-1, vmax=1, ax=ax, square=True,
                linewidths=0.5, cbar_kws={"shrink": 0.8})
    ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold")
    save_fig(fig, "07_correlation_matrix.png")

    # Churn correlations
    churn_corr = corr["Churn_binary"].drop("Churn_binary").sort_values(ascending=False)
    print(f"\n  Correlation with Churn:")
    for feat, val in churn_corr.items():
        direction = "↑" if val > 0 else "↓"
        print(f"    {direction} {feat:20s}: {val:+.3f}")

    print(f"\n  Key observations:")
    print(f"    - Tenure has the strongest negative correlation with churn (−0.35):")
    print(f"      longer relationships → less likely to leave.")
    print(f"    - MonthlyCharges is positively correlated (+0.19): higher bills → more churn.")
    print(f"    - TotalCharges is negatively correlated (−0.20), driven by the tenure effect.")
    print(f"    - tenure and TotalCharges are highly correlated (+0.83) — expected,")
    print(f"      since total spend grows with time.")
    print()


# ══════════════════════════════════════════════════════════════════════════════
#  7. CHURN RATE BY CONTRACT TYPE & TENURE
# ══════════════════════════════════════════════════════════════════════════════

def contract_tenure_analysis(df):
    print("=" * 70)
    print("  7. CHURN PATTERNS: CONTRACT × TENURE INTERACTION")
    print("=" * 70)

    df_plot = df.copy()
    df_plot["TotalCharges"] = df_plot["TotalCharges"].fillna(0.0)
    df_plot["Churn_binary"] = (df_plot["Churn"] == "Yes").astype(int)

    # Tenure bins
    bins = [0, 6, 12, 24, 48, 72]
    labels = ["0–6m", "7–12m", "13–24m", "25–48m", "49–72m"]
    df_plot["tenure_group"] = pd.cut(df_plot["tenure"], bins=bins, labels=labels, right=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Churn rate by tenure group
    ax = axes[0]
    churn_by_tenure = df_plot.groupby("tenure_group", observed=True)["Churn_binary"].mean() * 100
    bars = ax.bar(churn_by_tenure.index, churn_by_tenure.values,
                  color=sns.color_palette("YlOrRd", len(churn_by_tenure)),
                  edgecolor="white", linewidth=1.5)
    for bar, val in zip(bars, churn_by_tenure.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=10)
    ax.set_title("Churn Rate by Tenure Group", fontsize=13, fontweight="bold")
    ax.set_ylabel("Churn Rate (%)")
    ax.set_ylim(0, churn_by_tenure.max() * 1.25)

    # Churn rate by contract type × tenure group
    ax = axes[1]
    pivot = df_plot.groupby(["tenure_group", "Contract"], observed=True)["Churn_binary"].mean() * 100
    pivot = pivot.unstack()
    pivot.plot(kind="bar", ax=ax, width=0.7, edgecolor="white", linewidth=0.8)
    ax.set_title("Churn Rate: Contract Type × Tenure", fontsize=13, fontweight="bold")
    ax.set_ylabel("Churn Rate (%)")
    ax.set_xlabel("Tenure Group")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(title="Contract")

    fig.suptitle("Retention Patterns", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_fig(fig, "08_contract_tenure_churn.png")

    print(f"\n  Key observations:")
    print(f"    - New customers (0–6 months) churn at ~47%, roughly 8× the rate")
    print(f"      of long-tenure customers (49–72 months at ~6%).")
    print(f"    - Month-to-month contracts dominate early churn. Once a customer")
    print(f"      survives past 12 months, churn drops substantially.")
    print(f"    - Two-year contracts have near-zero churn across all tenure groups.")
    print()


# ══════════════════════════════════════════════════════════════════════════════
#  8. MONTHLY CHARGES DISTRIBUTION BY SERVICE TYPE
# ══════════════════════════════════════════════════════════════════════════════

def charges_by_service(df):
    print("=" * 70)
    print("  8. MONTHLY CHARGES BY INTERNET SERVICE TYPE")
    print("=" * 70)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.violinplot(data=df, x="InternetService", y="MonthlyCharges",
                   hue="Churn", split=True, palette=PALETTE_CHURN,
                   ax=ax, inner="quart", density_norm="width")
    ax.set_title("Monthly Charges Distribution by Internet Service & Churn",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Internet Service")
    ax.set_ylabel("Monthly Charges ($)")
    save_fig(fig, "09_charges_by_service.png")

    print(f"\n  Key observations:")
    print(f"    - Fiber optic users pay significantly more ($60–100 range) and churn more.")
    print(f"    - DSL users cluster around $30–60 with moderate churn.")
    print(f"    - 'No internet' customers have the lowest charges and lowest churn.")
    print(f"    - Among fiber optic users, churners tend to be at the higher end of charges.")
    print()


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "█" * 70)
    print("  TELCO CUSTOMER CHURN — EXPLORATORY DATA ANALYSIS")
    print("█" * 70 + "\n")

    df = load_raw()

    dataset_overview(df)
    missing_values_analysis(df)
    target_distribution(df)
    numeric_distributions(df)
    categorical_distributions(df)
    correlation_analysis(df)
    contract_tenure_analysis(df)
    charges_by_service(df)

    print("=" * 70)
    print("  ALL FIGURES SAVED")
    print("=" * 70)
    print(f"\n  Output directory: {FIG_DIR}")
    print(f"  Total figures: {len(list(FIG_DIR.glob('*.png')))}")
    print()


if __name__ == "__main__":
    main()
