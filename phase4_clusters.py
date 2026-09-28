# ============================================================
# ShopSurvival — Phase 4: Business Clusters (Unsupervised + Deep Learning)
# Topics: KMeans, Elbow method, Hierarchical clustering, PCA, ANN
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.cluster.hierarchy import dendrogram, linkage
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("data/cleaned_businesses.csv")

# Prepare numeric features for clustering
le = LabelEncoder()
df["category_enc"] = le.fit_transform(df["primary_category"].astype(str))
df["city_enc"]     = le.fit_transform(df["city"].astype(str))

feature_cols = ["stars", "log_reviews", "competition_density",
                "stars_zscore", "rating_bucket", "category_enc"]

X = df[feature_cols].fillna(0)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# ── 1. ELBOW METHOD ──────────────────────────────────────────
def elbow_method(X_scaled):
    inertias = []
    K_range = range(2, 12)
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(K_range, inertias, "bo-", markersize=8)
    ax.set_xlabel("Number of clusters (K)")
    ax.set_ylabel("Inertia (within-cluster sum of squares)")
    ax.set_title("Elbow method — finding the optimal K", fontsize=13)
    ax.axvline(x=4, color="red", linestyle="--", label="Optimal K=4")
    ax.legend()
    plt.tight_layout()
    plt.savefig("data/plot8_elbow.png", dpi=120)
    plt.show()
    print("✅ Saved plot8_elbow.png")
    return 4   # optimal K


# ── 2. KMEANS CLUSTERING ──────────────────────────────────────
CLUSTER_NAMES = {
    0: "Struggling Shops",
    1: "Steady Locals",
    2: "Popular Hubs",
    3: "Competitive Hotspots",
}

def run_kmeans(X_scaled, df, k=4):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    df["cluster"] = km.fit_predict(X_scaled)
    df["cluster_name"] = df["cluster"].map(CLUSTER_NAMES)

    print("\n📊 Cluster profiles:")
    profile = df.groupby("cluster_name")[
        ["stars", "review_count", "competition_density", "survived"]
    ].mean().round(2)
    profile["survived"] = (profile["survived"] * 100).round(1).astype(str) + "%"
    print(profile.to_string())
    return df, km


# ── 3. PCA — 2D VISUALISATION ────────────────────────────────
def pca_visualise(X_scaled, df):
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_scaled)
    explained = pca.explained_variance_ratio_ * 100

    print(f"\n🔍 PCA: 2 components explain {sum(explained):.1f}% of variance")
    print(f"   PC1: {explained[0]:.1f}%  |  PC2: {explained[1]:.1f}%")

    colors = ["#F0997B", "#5DCAA5", "#AFA9EC", "#FAC775"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Coloured by cluster
    for cid, name in CLUSTER_NAMES.items():
        mask = df["cluster"] == cid
        axes[0].scatter(X_2d[mask, 0], X_2d[mask, 1],
                        c=colors[cid], label=name, alpha=0.5, s=15)
    axes[0].set_xlabel(f"PC1 ({explained[0]:.1f}%)")
    axes[0].set_ylabel(f"PC2 ({explained[1]:.1f}%)")
    axes[0].set_title("PCA clusters — business personality types")
    axes[0].legend(fontsize=8)

    # Coloured by survival
    survived_mask = df["survived"] == 1
    axes[1].scatter(X_2d[~survived_mask, 0], X_2d[~survived_mask, 1],
                    c="#F0997B", label="Closed", alpha=0.4, s=10)
    axes[1].scatter(X_2d[survived_mask, 0], X_2d[survived_mask, 1],
                    c="#5DCAA5", label="Survived", alpha=0.4, s=10)
    axes[1].set_xlabel(f"PC1 ({explained[0]:.1f}%)")
    axes[1].set_ylabel(f"PC2 ({explained[1]:.1f}%)")
    axes[1].set_title("PCA — survived (green) vs closed (red)")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("data/plot9_pca_clusters.png", dpi=120)
    plt.show()
    print("✅ Saved plot9_pca_clusters.png")
    return pca, X_2d


# ── 4. HIERARCHICAL CLUSTERING — DENDROGRAM ──────────────────
def hierarchical_clustering(X_scaled, df, n=100):
    sample = X_scaled[:n]
    Z = linkage(sample, method="ward")

    fig, ax = plt.subplots(figsize=(14, 5))
    dendrogram(Z, ax=ax, leaf_rotation=90, leaf_font_size=6,
               color_threshold=0.7 * max(Z[:, 2]))
    ax.set_title("Hierarchical clustering dendrogram (sample of 100 businesses)", fontsize=13)
    ax.set_xlabel("Business index")
    ax.set_ylabel("Distance")
    plt.tight_layout()
    plt.savefig("data/plot10_dendrogram.png", dpi=120)
    plt.show()
    print("✅ Saved plot10_dendrogram.png")


# ── 5. ANN — DEEP LEARNING SURVIVAL PREDICTOR ────────────────
def train_ann(X_scaled, df):
    y = df["survived"]
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y)

    # MLPClassifier = ANN with configurable hidden layers
    ann = MLPClassifier(
        hidden_layer_sizes=(64, 32, 16),  # 3 hidden layers
        activation="relu",                 # ReLU activation
        solver="adam",                     # backpropagation optimizer
        max_iter=200,
        random_state=42,
        verbose=False,
        early_stopping=True,
        validation_fraction=0.1,
    )
    ann.fit(X_train, y_train)
    y_pred = ann.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n🧠 ANN (3 hidden layers: 64→32→16) accuracy: {acc*100:.1f}%")
    print(f"   Architecture: input({X_scaled.shape[1]}) → 64 → 32 → 16 → output(2)")
    print(f"   Activation: ReLU  |  Optimizer: Adam (backpropagation)")

    # Plot loss curve
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(ann.loss_curve_, color="#534AB7", label="Training loss")
    if ann.validation_scores_ is not None:
        val_loss = [1 - s for s in ann.validation_scores_]
        ax.plot(val_loss, color="#F0997B", label="Validation loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("ANN training — loss curve (backpropagation)")
    ax.legend()
    plt.tight_layout()
    plt.savefig("data/plot11_ann_loss.png", dpi=120)
    plt.show()
    print("✅ Saved plot11_ann_loss.png")
    return ann


# ── MAIN ─────────────────────────────────────────────────────
if __name__ == "__main__":
    k = elbow_method(X_scaled)
    df, km = run_kmeans(X_scaled, df, k)
    pca, X_2d = pca_visualise(X_scaled, df)
    hierarchical_clustering(X_scaled, df)
    ann = train_ann(X_scaled, df)

    # Save for app
    import joblib
    joblib.dump(km,  "models/kmeans.pkl")
    joblib.dump(pca, "models/pca.pkl")
    joblib.dump(ann, "models/ann.pkl")
    df.to_csv("data/clustered_businesses.csv", index=False)
    print("\n✅ Phase 4 complete — models and data saved")
