from __future__ import annotations

from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
REPORT = ROOT / "eda_report.txt"

# The only network/cache load in this module.
df = sns.load_dataset("titanic")
df.to_csv(ROOT / "titanic.csv", index=False)

with REPORT.open("w", encoding="utf-8") as f:
    f.write(f"shape={df.shape}\n\n")
    f.write("INFO\n")
    df.info(buf=f)
    f.write("\nDESCRIBE\n")
    f.write(df.describe(include="all").to_string() + "\n\n")

    missing = df.isna().mean().mul(100).loc[lambda s: s > 0].sort_values(ascending=False)
    f.write("MISSING PERCENTAGES\n")
    f.write(missing.to_string() + "\n\n")

    cleaned = df.copy()

    # Assignment threshold:
    # <5% -> drop rows; 5-30% -> impute; very high -> drop column.
    high_missing = [c for c in cleaned.columns if cleaned[c].isna().mean() > 0.30]
    cleaned = cleaned.drop(columns=high_missing)

    low_missing = [
        c for c in cleaned.columns
        if 0 < cleaned[c].isna().mean() < 0.05
    ]
    if low_missing:
        cleaned = cleaned.dropna(subset=low_missing)

    medium_missing = [
        c for c in cleaned.columns
        if 0.05 <= cleaned[c].isna().mean() <= 0.30
    ]
    for c in medium_missing:
        if pd.api.types.is_numeric_dtype(cleaned[c]):
            cleaned[c] = cleaned[c].fillna(cleaned[c].median())
        else:
            cleaned[c] = cleaned[c].fillna(cleaned[c].mode(dropna=True)[0])

    # Save the cleaned EDA data separately; the original titanic.csv remains the
    # required offline fallback.
    cleaned.to_csv(ROOT / "titanic_cleaned.csv", index=False)

    # Univariate plots.
    for col in ["age", "fare"]:
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.histplot(cleaned[col], kde=True, ax=ax)
        ax.set_title(f"{col.title()} distribution")
        fig.tight_layout()
        fig.savefig(FIG / f"{col}_hist.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 3))
        sns.boxplot(x=cleaned[col], ax=ax)
        ax.set_title(f"{col.title()} box plot")
        fig.tight_layout()
        fig.savefig(FIG / f"{col}_box.png", dpi=150)
        plt.close(fig)

    outlier_counts = {}
    for col in ["age", "fare"]:
        q1, q3 = cleaned[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outlier_counts[col] = int(((cleaned[col] < lo) | (cleaned[col] > hi)).sum())

    fare_mean = cleaned["fare"].mean()
    fare_median = cleaned["fare"].median()
    fare_mode = cleaned["fare"].mode().iloc[0]
    if fare_mean > fare_median > fare_mode:
        skew_text = "right-skewed"
    elif fare_mean < fare_median < fare_mode:
        skew_text = "left-skewed"
    else:
        skew_text = "not cleanly classified by mean/median/mode ordering"

    f.write("\nIQR OUTLIERS\n")
    f.write(str(outlier_counts) + "\n")
    f.write(f"\nFare mean={fare_mean:.4f}, median={fare_median:.4f}, mode={fare_mode:.4f}\n")
    f.write(f"Fare skew conclusion={skew_text}\n")

    # Bivariate survival rates.
    f.write("\nSURVIVAL RATE BY SEX\n")
    f.write((cleaned.groupby("sex")["survived"].mean().mul(100)).to_string() + "\n")
    f.write("\nSURVIVAL RATE BY PCLASS\n")
    f.write((cleaned.groupby("pclass")["survived"].mean().mul(100)).to_string() + "\n")
    f.write("\nSURVIVAL RATE BY SEX + PCLASS\n")
    f.write((cleaned.groupby(["sex", "pclass"])["survived"].mean().mul(100)).to_string() + "\n")

    corr_cols = ["survived", "pclass", "age", "sibsp", "parch", "fare"]
    corr = cleaned[corr_cols].corr()
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", ax=ax)
    ax.set_title("Required 6×6 correlation matrix")
    fig.tight_layout()
    fig.savefig(FIG / "correlation_heatmap.png", dpi=150)
    plt.close(fig)

    pairs = []
    for i, a in enumerate(corr_cols):
        for b in corr_cols[i + 1:]:
            pairs.append((a, b, corr.loc[a, b], abs(corr.loc[a, b])))
    top2 = sorted(pairs, key=lambda x: x[3], reverse=True)[:2]
    f.write("\nTWO STRONGEST ABSOLUTE CORRELATIONS\n")
    for a, b, value, _ in top2:
        f.write(f"{a} vs {b}: {value:.4f}\n")

    # Four multivariate charts with 2–4 sentence interpretations each.
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=cleaned, x="sex", y="survived", hue="pclass", ax=ax)
    ax.set_ylabel("Survival rate")
    ax.set_title("Survival by sex and passenger class")
    fig.tight_layout()
    fig.savefig(FIG / "survival_sex_pclass.png", dpi=150)
    plt.close(fig)
    f.write("\nCHART 1 INTERPRETATION\n")
    f.write("Survival rates vary across both sex and passenger class. "
            "The grouped bars make the interaction visible rather than treating either variable in isolation.\n")

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.boxplot(data=cleaned, x="pclass", y="age", hue="sex", ax=ax)
    ax.set_title("Age distribution by class and sex")
    fig.tight_layout()
    fig.savefig(FIG / "age_class_sex.png", dpi=150)
    plt.close(fig)
    f.write("\nCHART 2 INTERPRETATION\n")
    f.write("Age distributions differ across passenger classes and sexes. "
            "This provides demographic context for the survival patterns observed above.\n")

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.scatterplot(data=cleaned, x="age", y="fare", hue="survived", alpha=0.65, ax=ax)
    ax.set_title("Age vs fare by survival outcome")
    fig.tight_layout()
    fig.savefig(FIG / "age_fare_survival.png", dpi=150)
    plt.close(fig)
    f.write("\nCHART 3 INTERPRETATION\n")
    f.write("Fare and age occupy broad ranges, with survival outcomes distributed across the feature space. "
            "The plot is exploratory and should not be read as a causal relationship.\n")

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.pointplot(data=cleaned, x="pclass", y="survived", hue="sex", ax=ax)
    ax.set_ylabel("Survival rate")
    ax.set_title("Survival interaction")
    fig.tight_layout()
    fig.savefig(FIG / "survival_interaction.png", dpi=150)
    plt.close(fig)
    f.write("\nCHART 4 INTERPRETATION\n")
    f.write("The point plot summarizes the combined relationship of class and sex with survival. "
            "The separation between groups motivates including both variables in the predictive model.\n")

    # Exploratory z-score check on full cleaned EDA data.
    scaler = StandardScaler()
    z = scaler.fit_transform(cleaned[["age", "fare"]])
    zdf = pd.DataFrame(z, columns=["age_z", "fare_z"])
    f.write("\nSTANDARDIZATION CHECK\n")
    f.write(zdf.agg(["mean", "std"]).to_string() + "\n")

print(f"Wrote {REPORT}")
