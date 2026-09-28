# ============================================================
# ShopSurvival — Phase 1: Data Engine
# Topics: Python basics, Data structures, Functions, NumPy, Pandas
# ============================================================

import pandas as pd
import numpy as np
import json
import os

# ── 1. LOAD DATA ─────────────────────────────────────────────
# Download from: https://www.kaggle.com/datasets/yelp-dataset/yelp-dataset
# File needed: yelp_academic_dataset_business.json
# Place it in the same folder as this script

def load_yelp_data(filepath="yelp_academic_dataset_business.json", nrows=50000):
    """Load Yelp business JSON into a Pandas DataFrame."""
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= nrows:
                break
            records.append(json.loads(line))
    df = pd.DataFrame(records)
    print(f"✅ Loaded {len(df)} businesses")
    return df


# ── 2. CLEAN DATA ────────────────────────────────────────────
def clean_data(df):
    """Clean nulls, fix types, create target column."""

    # Drop columns we don't need
    drop_cols = ["attributes", "hours", "address", "neighborhood"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Fix nulls
    df["stars"] = df["stars"].fillna(df["stars"].median())
    df["review_count"] = df["review_count"].fillna(0)
    df["city"] = df["city"].fillna("Unknown")
    df["state"] = df["state"].fillna("Unknown")

    # Remove duplicates
    df = df.drop_duplicates(subset="business_id")

    # Target variable: is_open → 1 = survived, 0 = closed
    df["survived"] = df["is_open"].astype(int)

    # Categories: keep first category only
    df["primary_category"] = df["categories"].apply(
        lambda x: x.split(",")[0].strip() if isinstance(x, str) else "Unknown"
    )

    print(f"✅ After cleaning: {len(df)} rows, {df.shape[1]} columns")
    print(f"   Survived: {df['survived'].sum()} | Closed: {(df['survived']==0).sum()}")
    return df


# ── 3. FEATURE ENGINEERING WITH NUMPY ───────────────────────
def engineer_features(df):
    """Create useful numeric features using NumPy."""

    # Competition density: how many businesses in same city+category?
    counts = df.groupby(["city", "primary_category"])["business_id"].transform("count")
    df["competition_density"] = counts.values

    # Star rating buckets using NumPy digitize
    bins = np.array([0, 2.5, 3.5, 4.5, 5.0])
    df["rating_bucket"] = np.digitize(df["stars"].values, bins)

    # Log of review count (reduces skew)
    df["log_reviews"] = np.log1p(df["review_count"].values)

    # Z-score of stars within category (how good vs peers?)
    mean_stars = df.groupby("primary_category")["stars"].transform("mean")
    std_stars  = df.groupby("primary_category")["stars"].transform("std").replace(0, 1)
    df["stars_zscore"] = ((df["stars"] - mean_stars) / std_stars).round(3)

    print("✅ Features engineered: competition_density, rating_bucket, log_reviews, stars_zscore")
    return df


# ── 4. REUSABLE FUNCTIONS WITH *args / **kwargs ──────────────
def top_businesses(*categories, city=None, min_stars=4.0, n=10, df=None):
    """
    Get top surviving businesses.
    Usage: top_businesses('Restaurants', 'Pizza', city='Phoenix', n=5, df=df)
    """
    result = df[df["survived"] == 1].copy()
    if categories:
        mask = result["primary_category"].isin(categories)
        result = result[mask]
    if city:
        result = result[result["city"].str.lower() == city.lower()]
    result = result[result["stars"] >= min_stars]
    return result.nlargest(n, "review_count")[
        ["name", "city", "primary_category", "stars", "review_count"]
    ]


def survival_rate_by(**groupby_kwargs):
    """
    Get survival rate grouped by any column.
    Usage: survival_rate_by(col='primary_category', df=df, top_n=10)
    """
    col  = groupby_kwargs.get("col", "primary_category")
    data = groupby_kwargs.get("df")
    n    = groupby_kwargs.get("top_n", 10)

    result = (
        data.groupby(col)["survived"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "survival_rate", "count": "total"})
        .query("total >= 50")          # only categories with enough data
        .sort_values("survival_rate", ascending=False)
        .head(n)
    )
    result["survival_rate"] = (result["survival_rate"] * 100).round(1)
    return result


# ── 5. GROUPBY AGGREGATIONS ──────────────────────────────────
def summary_stats(df):
    """Print key GroupBy insights."""

    print("\n📊 Survival rate by state (top 5):")
    by_state = survival_rate_by(col="state", df=df, top_n=5)
    print(by_state.to_string())

    print("\n📊 Survival rate by category (top 10):")
    by_cat = survival_rate_by(col="primary_category", df=df, top_n=10)
    print(by_cat.to_string())

    print("\n📊 Avg stars of survived vs closed:")
    print(df.groupby("survived")["stars"].agg(["mean", "std"]).round(3))

    print("\n📊 Top open businesses in Restaurants:")
    print(top_businesses("Restaurants", min_stars=4.5, n=5, df=df).to_string(index=False))


# ── MAIN ─────────────────────────────────────────────────────
if __name__ == "__main__":
    # OPTION A: use real Yelp data
    # df = load_yelp_data("yelp_academic_dataset_business.json")

    # OPTION B: generate mock data to test pipeline (no download needed)
    np.random.seed(42)
    n = 2000
    categories = ["Restaurants", "Shopping", "Beauty & Spas", "Health", "Automotive",
                  "Pizza", "Coffee & Tea", "Hotels", "Bars", "Retail"]
    cities = ["Phoenix", "Las Vegas", "Charlotte", "Pittsburgh", "Cleveland"]

    df_mock = pd.DataFrame({
        "business_id":    [f"biz_{i}" for i in range(n)],
        "name":           [f"Business {i}" for i in range(n)],
        "city":           np.random.choice(cities, n),
        "state":          np.random.choice(["AZ", "NV", "NC", "PA", "OH"], n),
        "stars":          np.round(np.random.uniform(1, 5, n) * 2) / 2,
        "review_count":   np.random.randint(1, 2000, n),
        "is_open":        np.random.choice([0, 1], n, p=[0.2, 0.8]),
        "categories":     np.random.choice(categories, n),
    })

    df_mock = clean_data(df_mock)
    df_mock = engineer_features(df_mock)
    summary_stats(df_mock)

    # Save cleaned data for next phases
    os.makedirs("data", exist_ok=True)
    df_mock.to_csv("data/cleaned_businesses.csv", index=False)
    print("\n✅ Saved → data/cleaned_businesses.csv")
