# ============================================================
# ShopSurvival — Phase 3: Prediction Engine (ML Models)
# Topics: Hypothesis testing, Logistic/Linear Regression,
#         Decision Tree, Random Forest, SVM, Naive Bayes, KNN
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, mean_squared_error, r2_score)
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("data/cleaned_businesses.csv")


# ── 1. HYPOTHESIS TEST ───────────────────────────────────────
def hypothesis_test(df):
    """
    H0: Stars of survived and closed businesses are equal
    H1: Survived businesses have significantly higher stars
    """
    survived_stars = df[df["survived"] == 1]["stars"]
    closed_stars   = df[df["survived"] == 0]["stars"]

    t_stat, p_value = stats.ttest_ind(survived_stars, closed_stars)

    print("=" * 50)
    print("HYPOTHESIS TEST: Do stars predict survival?")
    print("=" * 50)
    print(f"  H0: Mean stars are equal for open vs closed")
    print(f"  H1: Survived businesses have higher mean stars")
    print(f"  T-statistic : {t_stat:.4f}")
    print(f"  P-value     : {p_value:.6f}")
    alpha = 0.05
    if p_value < alpha:
        print(f"  ✅ Reject H0 (p < {alpha}) — stars DO significantly affect survival!")
    else:
        print(f"  ❌ Fail to reject H0 — no significant difference")
    print()


# ── 2. PREPARE FEATURES ──────────────────────────────────────
def prepare_features(df):
    le = LabelEncoder()
    df = df.copy()
    df["category_enc"] = le.fit_transform(df["primary_category"].astype(str))
    df["city_enc"]     = le.fit_transform(df["city"].astype(str))

    feature_cols = ["stars", "log_reviews", "competition_density",
                    "stars_zscore", "rating_bucket", "category_enc", "city_enc"]

    X = df[feature_cols].fillna(0)
    y = df["survived"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, X_train_sc, X_test_sc, feature_cols, scaler


# ── 3. TRAIN ALL MODELS ──────────────────────────────────────
def train_all_models(X_train, X_test, y_train, y_test,
                     X_train_sc, X_test_sc, feature_cols):
    results = {}

    models = {
        "Logistic Regression": (LogisticRegression(max_iter=500), X_train_sc, X_test_sc),
        "Decision Tree":       (DecisionTreeClassifier(max_depth=5, random_state=42), X_train, X_test),
        "Random Forest":       (RandomForestClassifier(n_estimators=100, random_state=42), X_train, X_test),
        "SVM":                 (SVC(kernel="rbf", probability=True, random_state=42), X_train_sc, X_test_sc),
        "Naive Bayes":         (GaussianNB(), X_train_sc, X_test_sc),
        "KNN":                 (KNeighborsClassifier(n_neighbors=5), X_train_sc, X_test_sc),
    }

    print("=" * 55)
    print(f"{'Model':<22} {'Accuracy':>10} {'Precision':>10} {'Recall':>10}")
    print("=" * 55)

    for name, (model, Xtr, Xte) in models.items():
        model.fit(Xtr, y_train)
        y_pred = model.predict(Xte)
        acc  = accuracy_score(y_test, y_pred)
        rep  = classification_report(y_test, y_pred, output_dict=True)
        prec = rep["weighted avg"]["precision"]
        rec  = rep["weighted avg"]["recall"]
        results[name] = {"model": model, "accuracy": acc,
                         "precision": prec, "recall": rec, "y_pred": y_pred}
        print(f"{name:<22} {acc*100:>9.1f}% {prec*100:>9.1f}% {rec*100:>9.1f}%")

    print("=" * 55)

    # Best model
    best = max(results, key=lambda k: results[k]["accuracy"])
    print(f"\n🏆 Best model: {best} ({results[best]['accuracy']*100:.1f}% accuracy)")
    return results, best


# ── 4. CONFUSION MATRIX ──────────────────────────────────────
def plot_confusion_matrix(y_test, y_pred, model_name):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Closed", "Survived"],
                yticklabels=["Closed", "Survived"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_name}")
    plt.tight_layout()
    plt.savefig("data/plot6_confusion_matrix.png", dpi=120)
    plt.show()
    print("✅ Saved plot6_confusion_matrix.png")


# ── 5. FEATURE IMPORTANCE (Random Forest) ───────────────────
def plot_feature_importance(results, feature_cols):
    rf = results["Random Forest"]["model"]
    importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values()

    fig, ax = plt.subplots(figsize=(8, 4))
    importances.plot(kind="barh", ax=ax, color="#378ADD")
    ax.set_title("What matters most for business survival? (Random Forest)", fontsize=12)
    ax.set_xlabel("Feature Importance")
    plt.tight_layout()
    plt.savefig("data/plot7_feature_importance.png", dpi=120)
    plt.show()
    print("✅ Saved plot7_feature_importance.png")

    print("\n📊 Top 3 survival factors:")
    for feat, val in importances.sort_values(ascending=False).head(3).items():
        print(f"   {feat}: {val*100:.1f}%")


# ── 6. DECISION TREE RULES (human readable) ──────────────────
def print_tree_rules(results, feature_cols):
    dt = results["Decision Tree"]["model"]
    rules = export_text(dt, feature_names=feature_cols, max_depth=3)
    print("\n🌲 Decision Tree Rules (top 3 levels):")
    print(rules)


# ── 7. LINEAR REGRESSION (predict stars, for regression topic) ─
def linear_regression_demo(df):
    """Predict star rating as a regression task (continuous output)."""
    X = df[["log_reviews", "competition_density"]].fillna(0)
    y = df["stars"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse  = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_test, y_pred)

    print("\n📈 Linear Regression — predict star rating:")
    print(f"   MSE  : {mse:.4f}")
    print(f"   RMSE : {rmse:.4f}  (avg error in stars)")
    print(f"   R²   : {r2:.4f}  (how much variance explained)")


# ── MAIN ─────────────────────────────────────────────────────
if __name__ == "__main__":
    hypothesis_test(df)

    X_train, X_test, y_train, y_test, \
    X_train_sc, X_test_sc, feature_cols, scaler = prepare_features(df)

    results, best = train_all_models(
        X_train, X_test, y_train, y_test,
        X_train_sc, X_test_sc, feature_cols)

    plot_confusion_matrix(y_test, results[best]["y_pred"], best)
    plot_feature_importance(results, feature_cols)
    print_tree_rules(results, feature_cols)
    linear_regression_demo(df)

    # Save best model for app
    import joblib, os
    os.makedirs("models", exist_ok=True)
    joblib.dump(results["Random Forest"]["model"], "models/best_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    print("\n✅ Saved models/best_model.pkl and models/scaler.pkl")
