# ============================================================
# ShopSurvival — Phase 2: Flavor Dashboard (EDA + Visualisation)
# Topics: Matplotlib, Seaborn, EDA, Statistics (mean/std/kurtosis/covariance)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# Style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "#f9f9f9"})

df = pd.read_csv("data/cleaned_businesses.csv")


# ── 1. SURVIVAL RATE BY CATEGORY (Bar Chart) ─────────────────
def plot_survival_by_category(df):
    survival = (
        df.groupby("primary_category")["survived"]
        .agg(["mean", "count"])
        .query("count >= 30")
        .sort_values("mean", ascending=False)
        .head(12)
    )
    survival["mean"] *= 100

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(survival.index, survival["mean"],
                   color=["#5DCAA5" if v > 80 else "#F0997B" for v in survival["mean"]])
    ax.set_xlabel("Survival Rate (%)")
    ax.set_title("Which business categories survive best?", fontsize=14, fontweight="bold")
    ax.axvline(x=survival["mean"].mean(), color="gray", linestyle="--", label="Average")
    ax.legend()
    for bar, val in zip(bars, survival["mean"]):
        ax.text(val + 0.5, bar.get_y() + bar.get_height()/2,
                f"{val:.1f}%", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig("data/plot1_survival_by_category.png", dpi=120)
    plt.show()
    print("✅ Saved plot1_survival_by_category.png")


# ── 2. STAR RATING DISTRIBUTION + NORMALITY TEST ─────────────
def plot_rating_distribution(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Histogram + KDE
    for label, group in df.groupby("survived"):
        name = "Survived" if label else "Closed"
        color = "#5DCAA5" if label else "#F0997B"
        axes[0].hist(group["stars"], bins=20, alpha=0.6, label=name,
                     color=color, density=True)

    axes[0].set_title("Star rating distribution: survived vs closed")
    axes[0].set_xlabel("Stars")
    axes[0].legend()

    # Statistics printout
    for label in [0, 1]:
        g = df[df["survived"] == label]["stars"]
        name = "Survived" if label else "Closed"
        print(f"\n📐 {name}:")
        print(f"   Mean     : {g.mean():.3f}")
        print(f"   Std Dev  : {g.std():.3f}")
        print(f"   Kurtosis : {g.kurtosis():.3f}  (0=normal, +ve=peaked, -ve=flat)")
        print(f"   Skewness : {g.skew():.3f}")

    # Q-Q plot to check normality
    stats.probplot(df["stars"], dist="norm", plot=axes[1])
    axes[1].set_title("Q-Q plot: are ratings normally distributed?")

    plt.tight_layout()
    plt.savefig("data/plot2_rating_distribution.png", dpi=120)
    plt.show()
    print("\n✅ Saved plot2_rating_distribution.png")


# ── 3. COVARIANCE / CORRELATION HEATMAP ──────────────────────
def plot_correlation_heatmap(df):
    num_cols = ["stars", "review_count", "log_reviews",
                "competition_density", "stars_zscore", "survived"]
    corr = df[num_cols].corr()

    # Print covariance insight
    cov = df[["stars", "survived"]].cov()
    print(f"\n📐 Covariance (stars vs survived): {cov.iloc[0,1]:.4f}")
    print(f"   Interpretation: {'positive → higher stars = more likely to survive' if cov.iloc[0,1]>0 else 'negative'}")

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, square=True, ax=ax,
                linewidths=0.5, cbar_kws={"shrink": 0.8})
    ax.set_title("Correlation heatmap — what predicts survival?", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("data/plot3_correlation_heatmap.png", dpi=120)
    plt.show()
    print("✅ Saved plot3_correlation_heatmap.png")


# ── 4. COMPETITION DENSITY vs SURVIVAL (Seaborn Boxplot) ─────
def plot_competition_vs_survival(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Boxplot
    df["status"] = df["survived"].map({1: "Survived", 0: "Closed"})
    sns.boxplot(data=df, x="status", y="competition_density",
                palette={"Survived": "#5DCAA5", "Closed": "#F0997B"}, ax=axes[0])
    axes[0].set_title("Competition density: survived vs closed")
    axes[0].set_ylabel("No. of similar businesses in same city")

    # Pairplot-style scatter
    axes[1].scatter(df[df["survived"]==1]["log_reviews"],
                    df[df["survived"]==1]["stars"],
                    alpha=0.3, s=10, color="#5DCAA5", label="Survived")
    axes[1].scatter(df[df["survived"]==0]["log_reviews"],
                    df[df["survived"]==0]["stars"],
                    alpha=0.3, s=10, color="#F0997B", label="Closed")
    axes[1].set_xlabel("Log(review count)")
    axes[1].set_ylabel("Stars")
    axes[1].set_title("Reviews vs Stars — do they separate survived/closed?")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("data/plot4_competition_scatter.png", dpi=120)
    plt.show()
    print("✅ Saved plot4_competition_scatter.png")


# ── 5. SEABORN PAIRPLOT ───────────────────────────────────────
def plot_pairplot(df):
    sample = df.sample(min(500, len(df)), random_state=42)
    sample["status"] = sample["survived"].map({1: "Survived", 0: "Closed"})
    pair_cols = ["stars", "log_reviews", "competition_density", "stars_zscore", "status"]
    g = sns.pairplot(sample[pair_cols], hue="status",
                     palette={"Survived": "#5DCAA5", "Closed": "#F0997B"},
                     plot_kws={"alpha": 0.4, "s": 15}, diag_kind="kde")
    g.fig.suptitle("Pairplot: feature relationships coloured by survival", y=1.02, fontsize=13)
    plt.savefig("data/plot5_pairplot.png", dpi=100, bbox_inches="tight")
    plt.show()
    print("✅ Saved plot5_pairplot.png")


# ── MAIN ─────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Running EDA...\n")
    plot_survival_by_category(df)
    plot_rating_distribution(df)
    plot_correlation_heatmap(df)
    plot_competition_vs_survival(df)
    plot_pairplot(df)
    print("\n✅ Phase 2 complete — all plots saved to data/")
