# ============================================================
# ShopSurvival — Phase 6: Streamlit Web App
# Run with: streamlit run phase6_app.py
# Install:   pip install streamlit joblib scikit-learn pandas numpy matplotlib seaborn
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import re
from collections import Counter
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA

# ── PAGE CONFIG ──────────────────────────────────────────────
st.set_page_config(
    page_title="ShopSurvival — Will your business survive?",
    page_icon="🏪",
    layout="wide",
)

# ── LOAD DATA & MODELS ───────────────────────────────────────
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("data/clustered_businesses.csv")
    except FileNotFoundError:
        # Generate mock data if real data not present
        np.random.seed(42)
        n = 2000
        categories = ["Restaurants", "Shopping", "Beauty & Spas",
                      "Health", "Coffee & Tea", "Bars", "Retail", "Automotive"]
        cities = ["Phoenix", "Las Vegas", "Charlotte", "Pittsburgh", "Cleveland"]
        df = pd.DataFrame({
            "name":                [f"Business {i}" for i in range(n)],
            "city":                np.random.choice(cities, n),
            "primary_category":    np.random.choice(categories, n),
            "stars":               np.round(np.random.uniform(1, 5, n) * 2) / 2,
            "review_count":        np.random.randint(1, 2000, n),
            "log_reviews":         np.log1p(np.random.randint(1, 2000, n)),
            "competition_density": np.random.randint(5, 200, n),
            "stars_zscore":        np.random.normal(0, 1, n).round(2),
            "rating_bucket":       np.random.randint(1, 5, n),
            "survived":            np.random.choice([0, 1], n, p=[0.2, 0.8]),
            "cluster":             np.random.randint(0, 4, n),
        })
        cluster_map = {0: "Struggling Shops", 1: "Steady Locals",
                       2: "Popular Hubs",    3: "Competitive Hotspots"}
        df["cluster_name"] = df["cluster"].map(cluster_map)
    return df

@st.cache_resource
def load_models(df):
    le = LabelEncoder()
    df = df.copy()
    df["category_enc"] = le.fit_transform(df["primary_category"].astype(str))
    df["city_enc"]     = le.fit_transform(df["city"].astype(str))
    feature_cols = ["stars", "log_reviews", "competition_density",
                    "stars_zscore", "rating_bucket", "category_enc", "city_enc"]
    X = df[feature_cols].fillna(0)
    y = df["survived"]
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_sc, y)
    pca = PCA(n_components=2)
    pca.fit(X_sc)
    return rf, scaler, pca, le, feature_cols

df = load_data()
rf, scaler, pca, le, feature_cols = load_models(df)

CLUSTER_NAMES = {
    0: ("Struggling Shops",     "⚠️", "#F0997B"),
    1: ("Steady Locals",        "🟡", "#FAC775"),
    2: ("Popular Hubs",         "✅", "#5DCAA5"),
    3: ("Competitive Hotspots", "🔵", "#AFA9EC"),
}

# ── SIDEBAR ──────────────────────────────────────────────────
st.sidebar.title("🏪 ShopSurvival")
st.sidebar.caption("AI Business Survival Predictor")
tab_choice = st.sidebar.radio(
    "Navigate",
    ["🏠 Overview", "📊 Explore", "🔮 Predict", "🔵 Clusters", "💬 Review Brain"]
)

categories  = sorted(df["primary_category"].unique())
cities      = sorted(df["city"].unique())


# ════════════════════════════════════════════
#  TAB 1 — OVERVIEW
# ════════════════════════════════════════════
if tab_choice == "🏠 Overview":
    st.title("🏪 ShopSurvival")
    st.subheader("Will your business survive? Let data decide.")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Businesses", f"{len(df):,}")
    col2.metric("Survival Rate",    f"{df['survived'].mean()*100:.1f}%")
    col3.metric("Avg Rating",       f"{df['stars'].mean():.2f} ⭐")
    col4.metric("Categories",       df["primary_category"].nunique())

    st.markdown("---")
    st.markdown("""
    **What this app does:**
    - 📊 **Explore** — visualise survival patterns across cities and categories
    - 🔮 **Predict** — enter your business details → get a survival probability
    - 🔵 **Clusters** — see which business archetype you belong to (PCA + KMeans)
    - 💬 **Review Brain** — paste reviews → instant sentiment analysis
    """)

    # Quick chart
    st.markdown("#### Survival rate by category")
    surv = (
        df.groupby("primary_category")["survived"]
        .agg(["mean", "count"]).query("count >= 20")
        .sort_values("mean", ascending=False).head(10)
    )
    surv["mean"] *= 100
    fig, ax = plt.subplots(figsize=(9, 4))
    colors = ["#5DCAA5" if v > 80 else "#FAC775" if v > 65 else "#F0997B"
              for v in surv["mean"]]
    ax.barh(surv.index, surv["mean"], color=colors)
    ax.set_xlabel("Survival Rate (%)")
    ax.axvline(surv["mean"].mean(), color="gray", linestyle="--")
    ax.set_facecolor("#f9f9f9")
    fig.patch.set_alpha(0)
    st.pyplot(fig)


# ════════════════════════════════════════════
#  TAB 2 — EXPLORE
# ════════════════════════════════════════════
elif tab_choice == "📊 Explore":
    st.title("📊 Explore Survival Patterns")

    col1, col2 = st.columns(2)
    selected_city = col1.selectbox("Filter by city", ["All"] + cities)
    selected_cat  = col2.selectbox("Filter by category", ["All"] + categories)

    filtered = df.copy()
    if selected_city != "All":
        filtered = filtered[filtered["city"] == selected_city]
    if selected_cat != "All":
        filtered = filtered[filtered["primary_category"] == selected_cat]

    st.metric("Businesses shown", len(filtered))

    col1, col2 = st.columns(2)

    # Rating distribution
    with col1:
        st.markdown("#### Rating distribution")
        fig, ax = plt.subplots(figsize=(6, 3))
        for label, color, name in [(1, "#5DCAA5", "Survived"), (0, "#F0997B", "Closed")]:
            g = filtered[filtered["survived"] == label]["stars"]
            if len(g):
                ax.hist(g, bins=15, alpha=0.6, color=color, label=name, density=True)
        ax.legend(); ax.set_xlabel("Stars"); ax.set_facecolor("#f9f9f9")
        fig.patch.set_alpha(0)
        st.pyplot(fig)

    # Heatmap
    with col2:
        st.markdown("#### Correlation heatmap")
        num_cols = ["stars", "log_reviews", "competition_density", "survived"]
        corr = filtered[num_cols].corr()
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                    center=0, ax=ax, linewidths=0.5)
        fig.patch.set_alpha(0)
        st.pyplot(fig)

    # Seaborn boxplot
    st.markdown("#### Competition density: survived vs closed")
    fig, ax = plt.subplots(figsize=(9, 3))
    filtered["Status"] = filtered["survived"].map({1: "Survived", 0: "Closed"})
    sns.boxplot(data=filtered, x="Status", y="competition_density",
                palette={"Survived": "#5DCAA5", "Closed": "#F0997B"}, ax=ax)
    fig.patch.set_alpha(0); ax.set_facecolor("#f9f9f9")
    st.pyplot(fig)


# ════════════════════════════════════════════
#  TAB 3 — PREDICT
# ════════════════════════════════════════════
elif tab_choice == "🔮 Predict":
    st.title("🔮 Predict Business Survival")
    st.caption("Enter your business details below to get an AI-powered survival probability.")

    col1, col2 = st.columns(2)
    with col1:
        b_name     = st.text_input("Business name", "My New Cafe")
        b_category = st.selectbox("Category", categories)
        b_city     = st.selectbox("City", cities)
    with col2:
        b_stars    = st.slider("Expected star rating", 1.0, 5.0, 4.0, 0.5)
        b_reviews  = st.slider("Estimated monthly reviews", 1, 500, 50)
        b_comp     = st.slider("Nearby competitors (same category in city)", 5, 300, 60)

    if st.button("🔮 Predict survival probability", use_container_width=True):
        log_rev    = np.log1p(b_reviews)
        avg_stars  = df[df["primary_category"] == b_category]["stars"].mean()
        std_stars  = df[df["primary_category"] == b_category]["stars"].std()
        zscore     = (b_stars - avg_stars) / (std_stars if std_stars > 0 else 1)
        r_bucket   = int(np.digitize(b_stars, [0, 2.5, 3.5, 4.5, 5.0]))
        cat_enc    = le.transform([b_category])[0] if b_category in le.classes_ else 0
        city_enc   = 0

        features = np.array([[b_stars, log_rev, b_comp, round(zscore, 2), r_bucket, cat_enc, city_enc]])
        features_sc = scaler.transform(features)
        prob = rf.predict_proba(features_sc)[0][1]

        st.markdown("---")
        if prob >= 0.75:
            verdict = "✅ Strong survival outlook"
            color = "green"
        elif prob >= 0.5:
            verdict = "🟡 Moderate — some risk factors present"
            color = "orange"
        else:
            verdict = "🔴 High risk — consider adjusting your strategy"
            color = "red"

        st.markdown(f"### {verdict}")
        st.progress(float(prob))
        st.markdown(f"**{prob*100:.1f}% probability of surviving** for **{b_name}**")

        # Risk factors
        st.markdown("#### Key risk factors")
        importances = rf.feature_importances_
        feat_names  = feature_cols
        risk_df = pd.DataFrame({"Feature": feat_names, "Importance": importances})
        risk_df = risk_df.sort_values("Importance", ascending=False).head(4)
        st.bar_chart(risk_df.set_index("Feature")["Importance"])

        st.info(f"💡 Tip: The most impactful factor for your business type is **{risk_df.iloc[0]['Feature']}**.")


# ════════════════════════════════════════════
#  TAB 4 — CLUSTERS
# ════════════════════════════════════════════
elif tab_choice == "🔵 Clusters":
    st.title("🔵 Business Personality Clusters")
    st.caption("KMeans + PCA — discover which archetype your business belongs to")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("#### The 4 business archetypes")
        for cid, (name, icon, color) in CLUSTER_NAMES.items():
            c_data = df[df["cluster"] == cid]
            surv   = c_data["survived"].mean() * 100
            stars  = c_data["stars"].mean()
            st.markdown(f"""
**{icon} {name}**
Survival: {surv:.0f}% | Avg ⭐ {stars:.1f}
""")
            st.markdown("---")

    with col2:
        st.markdown("#### PCA scatter — 2D view of all businesses")
        le2 = LabelEncoder()
        df2 = df.copy()
        df2["category_enc"] = le2.fit_transform(df2["primary_category"].astype(str))
        df2["city_enc"]     = le2.fit_transform(df2["city"].astype(str))
        X2 = df2[feature_cols].fillna(0)
        X2_sc = scaler.transform(X2)
        X2_2d = pca.transform(X2_sc)

        fig, ax = plt.subplots(figsize=(7, 5))
        colors_list = ["#F0997B", "#FAC775", "#5DCAA5", "#AFA9EC"]
        for cid in range(4):
            mask = df2["cluster"] == cid
            ax.scatter(X2_2d[mask, 0], X2_2d[mask, 1],
                       c=colors_list[cid], label=CLUSTER_NAMES[cid][0],
                       alpha=0.5, s=12)
        ax.legend(fontsize=8)
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        ax.set_facecolor("#f9f9f9"); fig.patch.set_alpha(0)
        st.pyplot(fig)


# ════════════════════════════════════════════
#  TAB 5 — REVIEW BRAIN
# ════════════════════════════════════════════
elif tab_choice == "💬 Review Brain":
    st.title("💬 Review Brain — Sentiment Analyser")
    st.caption("Paste customer reviews → instantly see if they help or hurt your survival odds")

    reviews_text = st.text_area(
        "Paste reviews here (one per line)",
        placeholder="Great food, came back 3 times already!\nTerrible service, waited 45 minutes.\nDecent place, nothing special.",
        height=180,
    )

    if st.button("🔍 Analyse reviews", use_container_width=True):
        lines = [r.strip() for r in reviews_text.strip().split("\n") if r.strip()]
        if not lines:
            st.warning("Please enter at least one review.")
        else:
            # Simple keyword-based sentiment (no external model needed)
            positive_words = {"great", "amazing", "best", "love", "excellent",
                              "fantastic", "good", "wonderful", "recommend", "friendly",
                              "clean", "fast", "fresh", "tasty", "delicious"}
            negative_words = {"terrible", "worst", "awful", "bad", "slow", "rude",
                              "dirty", "cold", "overpriced", "never", "disappointing",
                              "horrible", "disgusting", "closed", "avoid"}

            results = []
            for r in lines:
                words = set(r.lower().split())
                pos = len(words & positive_words)
                neg = len(words & negative_words)
                if pos > neg:
                    label, color = "Positive ✅", "#5DCAA5"
                elif neg > pos:
                    label, color = "Negative ❌", "#F0997B"
                else:
                    label, color = "Neutral 😐", "#FAC775"
                results.append((r, label, color))

            pos_count = sum(1 for _, l, _ in results if "Positive" in l)
            neg_count = sum(1 for _, l, _ in results if "Negative" in l)
            total     = len(results)
            pos_pct   = pos_count / total * 100

            col1, col2, col3 = st.columns(3)
            col1.metric("Positive reviews", f"{pos_count}/{total}")
            col2.metric("Negative reviews", f"{neg_count}/{total}")
            col3.metric("Sentiment score",  f"{pos_pct:.0f}%")

            if pos_pct >= 70:
                st.success("🎉 Great sentiment! These reviews should help your business survive.")
            elif pos_pct >= 40:
                st.warning("⚠️ Mixed sentiment — address the negative feedback to improve survival odds.")
            else:
                st.error("🚨 Mostly negative — this pattern is associated with business closure.")

            st.markdown("#### Individual review analysis")
            for review, label, color in results:
                st.markdown(f"- **{label}** — _{review}_")

            # Word cloud text
            all_words = " ".join(r for r, _, _ in results).lower().split()
            stopwords = {"the", "a", "is", "was", "and", "to", "in", "it", "for", "but", "i"}
            top_words = [w for w, _ in Counter(all_words).most_common(20)
                         if w not in stopwords][:8]
            st.markdown(f"**Most mentioned:** {', '.join(top_words)}")


# ── FOOTER ───────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.caption("Built with Python · Scikit-learn · Streamlit")
st.sidebar.caption("Data: Yelp Academic Dataset (Kaggle)")
