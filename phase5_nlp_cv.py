# ============================================================
# ShopSurvival — Phase 5: Review Brain + Storefront Lens
# Topics: NLP text processing, Sentiment analysis, CNN, Computer Vision
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import re

# NLP
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings("ignore")


# ══════════════════════════════════════════
#  PART A — NLP: Review Brain
# ══════════════════════════════════════════

# Sample reviews (replace with real Yelp review data)
SAMPLE_REVIEWS = [
    ("Amazing food, fast service, will definitely come back!", 1),
    ("Terrible experience. Waited 40 mins, food was cold.", 0),
    ("Average place, nothing special but nothing bad either.", 1),
    ("Best biryani in the city! Staff is super friendly.", 1),
    ("Overpriced and rude staff. Never going again.", 0),
    ("Good ambiance but the food quality has declined.", 0),
    ("Hidden gem! Authentic flavours, highly recommend.", 1),
    ("Dirty tables, slow service. Very disappointing.", 0),
    ("Loved it! Great value for money.", 1),
    ("Food was okay but parking is a nightmare.", 1),
    ("Shut down within months — bad hygiene issues.", 0),
    ("Consistently great. My family eats here every week.", 1),
]


# ── 1. TEXT CLEANING ─────────────────────────────────────────
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)       # remove punctuation
    text = re.sub(r"\s+", " ", text).strip()    # remove extra spaces
    return text


# ── 2. WORD FREQUENCY ANALYSIS ───────────────────────────────
def word_frequency_analysis(reviews):
    survived_words, closed_words = [], []
    for text, label in reviews:
        words = clean_text(text).split()
        if label == 1:
            survived_words.extend(words)
        else:
            closed_words.extend(words)

    stopwords = {"the", "a", "is", "was", "and", "to", "in", "it",
                 "for", "but", "of", "my", "i", "be", "has", "are"}
    survived_top = [(w, c) for w, c in Counter(survived_words).most_common(15)
                    if w not in stopwords]
    closed_top   = [(w, c) for w, c in Counter(closed_words).most_common(15)
                    if w not in stopwords]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, data, color, title in [
        (axes[0], survived_top, "#5DCAA5", "Top words in SURVIVING business reviews"),
        (axes[1], closed_top,   "#F0997B", "Top words in CLOSED business reviews"),
    ]:
        words, counts = zip(*data)
        ax.barh(words, counts, color=color)
        ax.set_title(title, fontsize=11)
        ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig("data/plot12_word_freq.png", dpi=120)
    plt.show()
    print("✅ Saved plot12_word_freq.png")


# ── 3. TFIDF + NAIVE BAYES SENTIMENT CLASSIFIER ──────────────
def train_sentiment_classifier(reviews):
    texts  = [clean_text(r[0]) for r in reviews]
    labels = [r[1] for r in reviews]

    vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
    X = vectorizer.fit_transform(texts)

    # For small demo data, use full set; in real project split 80/20
    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.3, random_state=42)

    model = MultinomialNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n💬 Sentiment classifier (TF-IDF + Naive Bayes) accuracy: {acc*100:.1f}%")

    return model, vectorizer


# ── 4. AUTO-SUMMARY GENERATOR ────────────────────────────────
def generate_summary(business_name, reviews_list, model, vectorizer):
    """Generate a one-line taste summary from a list of reviews."""
    cleaned = [clean_text(r) for r in reviews_list]
    X = vectorizer.transform(cleaned)
    sentiments = model.predict(X)
    pos_pct = int(sentiments.mean() * 100)

    all_words = " ".join(cleaned).split()
    stopwords = {"the", "a", "is", "was", "and", "to", "in", "it", "for", "but", "of"}
    top_words = [w for w, _ in Counter(all_words).most_common(20) if w not in stopwords][:3]

    mood = "loved for" if pos_pct >= 60 else "criticised for"
    summary = (f"{business_name} is {mood} its {', '.join(top_words)} "
               f"({pos_pct}% positive reviews).")
    print(f"\n📝 Auto-summary: {summary}")
    return summary


# ══════════════════════════════════════════
#  PART B — CNN: Storefront Image Classifier
# ══════════════════════════════════════════

# ── 5. CNN WITH TENSORFLOW/KERAS ─────────────────────────────
def build_and_train_cnn():
    """
    Train a CNN to classify storefront images as:
      0 = Poor condition (likely to close)
      1 = Good condition (likely to survive)

    In a real project:
      - Collect ~100 images per class from Google Maps street view
      - Or use Food-101 / Open Images as proxy dataset
      - Place images in: images/good/ and images/poor/

    Below: a minimal runnable example with synthetic pixel data.
    """
    try:
        import tensorflow as tf
        from tensorflow.keras import layers, models
        from tensorflow.keras.preprocessing.image import ImageDataGenerator

        print("\n🖼 Building CNN for storefront image classification...")

        # ── Architecture ───────────────────────────────────────
        model = models.Sequential([
            # Block 1
            layers.Conv2D(32, (3, 3), activation="relu", input_shape=(64, 64, 3)),
            layers.MaxPooling2D(2, 2),

            # Block 2
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D(2, 2),

            # Block 3
            layers.Conv2D(128, (3, 3), activation="relu"),
            layers.MaxPooling2D(2, 2),

            # Classifier head
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(1, activation="sigmoid"),   # binary: good/poor
        ])

        model.compile(optimizer="adam",
                      loss="binary_crossentropy",
                      metrics=["accuracy"])

        model.summary()

        # ── Synthetic demo (replace with real image data) ──────
        np.random.seed(42)
        n_samples = 200
        X_img = np.random.rand(n_samples, 64, 64, 3).astype("float32")
        y_img = np.random.randint(0, 2, n_samples)

        # Data augmentation (applied during training)
        datagen = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
        )

        history = model.fit(
            datagen.flow(X_img, y_img, batch_size=16),
            epochs=5,
            validation_split=0.2,
            verbose=1,
        )

        # Plot training curve
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(history.history["accuracy"],     label="Train accuracy", color="#534AB7")
        ax.plot(history.history["val_accuracy"], label="Val accuracy",   color="#F0997B")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.set_title("CNN training — storefront image classifier")
        ax.legend()
        plt.tight_layout()
        plt.savefig("data/plot13_cnn_training.png", dpi=120)
        plt.show()
        print("✅ Saved plot13_cnn_training.png")

        model.save("models/cnn_storefront.h5")
        print("✅ Saved models/cnn_storefront.h5")
        return model

    except ImportError:
        print("⚠ TensorFlow not installed. Run: pip install tensorflow")
        print("  CNN architecture is defined above — install TF to train it.")
        return None


# ── 6. PREDICT FROM IMAGE ────────────────────────────────────
def predict_storefront(image_path, cnn_model):
    """Given an image path, predict good/poor storefront."""
    try:
        from PIL import Image
        img = Image.open(image_path).resize((64, 64))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        prob = cnn_model.predict(img_array)[0][0]
        label = "Good condition (likely to survive)" if prob > 0.5 else "Poor condition (risk of closure)"
        print(f"\n🖼 Storefront prediction: {label} ({prob*100:.1f}% confidence)")
        return label, prob
    except Exception as e:
        print(f"Image prediction error: {e}")


# ── MAIN ─────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("PART A — NLP: Review Brain")
    print("=" * 50)

    word_frequency_analysis(SAMPLE_REVIEWS)
    sentiment_model, vectorizer = train_sentiment_classifier(SAMPLE_REVIEWS)

    # Test auto-summary
    sample_reviews_for_biz = [
        "Great service and amazing food",
        "Always crowded, food is fantastic",
        "Best place in town, love the ambiance"
    ]
    generate_summary("Spice Garden", sample_reviews_for_biz, sentiment_model, vectorizer)

    # Test single review
    test_review = "The place looks run-down and service was terrible"
    X_test = vectorizer.transform([clean_text(test_review)])
    pred = sentiment_model.predict(X_test)[0]
    print(f"\n💬 Review: '{test_review}'")
    print(f"   Sentiment: {'Positive ✅' if pred == 1 else 'Negative ❌'}")

    print("\n" + "=" * 50)
    print("PART B — CNN: Storefront Image Classifier")
    print("=" * 50)
    build_and_train_cnn()

    print("\n✅ Phase 5 complete")
