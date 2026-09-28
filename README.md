# 🏪 ShopSurvival — AI Business Survival Predictor

> **An end-to-end Data Science and Machine Learning project that analyzes business characteristics, identifies survival patterns, segments businesses into archetypes, and predicts business survival probability through an interactive Streamlit application.**

---

## 📌 Overview

**ShopSurvival** is an end-to-end Data Science project built to understand the factors associated with business survival and closure.

The project combines:

* 🐍 Python
* 📊 Exploratory Data Analysis
* 📈 Statistical Analysis
* 🤖 Machine Learning
* 🔵 Clustering
* 📉 PCA
* 💬 NLP & Sentiment Analysis
* 🧠 Neural Network / CNN concepts
* 🌐 Streamlit

The project is organized into six phases, progressing from data preprocessing and exploratory analysis to machine learning, clustering, NLP, and finally an interactive web application.

---

## 🎯 Project Objectives

The main objectives of ShopSurvival are to:

* Analyze business survival patterns.
* Understand how ratings, reviews, competition, location, and categories relate to survival.
* Build machine learning models for survival prediction.
* Segment businesses into meaningful business archetypes.
* Reduce high-dimensional data using PCA.
* Analyze customer reviews using NLP and sentiment analysis.
* Provide an interactive interface for business survival exploration and prediction.

---

## 🚀 Key Features

### 📊 1. Exploratory Data Analysis

The project performs extensive EDA to understand business characteristics and survival patterns.

Visualizations include:

* Survival rate by business category
* Rating distribution
* Correlation heatmap
* Competition analysis
* Pair plots
* Feature distributions

---

### 🤖 2. Machine Learning

Multiple machine learning techniques are explored for business survival prediction, including:

* Linear Regression
* Logistic Regression
* Decision Trees
* Random Forest
* Support Vector Machine
* Naive Bayes
* K-Nearest Neighbors

Classification performance is evaluated using appropriate metrics such as:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

---

### 🔵 3. Business Clustering

Unsupervised learning is used to identify different business archetypes.

The project includes:

* K-Means clustering
* Elbow method
* PCA visualization
* Hierarchical clustering
* Dendrogram analysis

Businesses are grouped into categories such as:

| Archetype               | Description                                             |
| ----------------------- | ------------------------------------------------------- |
| ⚠️ Struggling Shops     | Businesses with weaker survival-related characteristics |
| 🟡 Steady Locals        | Businesses showing relatively stable characteristics    |
| ✅ Popular Hubs          | Businesses with stronger engagement indicators          |
| 🔵 Competitive Hotspots | Businesses operating in highly competitive environments |

---

### 📉 4. PCA Visualization

Principal Component Analysis (PCA) is used to transform multiple business features into a lower-dimensional representation.

This allows business clusters to be visualized in a two-dimensional space.

---

### 💬 5. NLP & Sentiment Analysis

The project also explores business review text using Natural Language Processing techniques.

The NLP pipeline includes concepts such as:

* Text preprocessing
* Tokenization
* TF-IDF
* Word-frequency analysis
* Sentiment analysis

This helps investigate the relationship between review sentiment and business outcomes.

---

### 🔮 6. Business Survival Prediction

The Streamlit application allows users to enter business information such as:

* Business name
* Business category
* City
* Expected rating
* Estimated reviews
* Nearby competition

The application then generates a **predicted survival probability** using a Random Forest classification model.

> **Note:** The prediction is a machine-learning estimate based on the project's dataset and features. It should not be interpreted as a guarantee of future business performance.

---

## 🌐 Streamlit Application

The final application contains several interactive sections:

### 🏠 Overview

Displays key statistics including:

* Total businesses
* Survival rate
* Average rating
* Number of categories

### 📊 Explore

Allows users to explore survival patterns by:

* City
* Business category
* Rating
* Competition density
* Correlations

### 🔮 Predict

Users can enter business characteristics and receive a model-generated survival probability.

### 🔵 Clusters

Explores business archetypes and their PCA-based representation.

### 💬 Review Brain

Allows review text to be analyzed using sentiment-analysis techniques.

---

## 🗂️ Project Structure

```text
ShopSurvival/
│
├── data/
│   ├── cleaned_businesses.csv
│   ├── clustered_businesses.csv
│   ├── plot1_survival_by_category.png
│   ├── plot2_rating_distribution.png
│   ├── plot3_correlation_heatmap.png
│   ├── plot4_competition_scatter.png
│   ├── plot5_pairplot.png
│   ├── plot6_confusion_matrix.png
│   ├── plot7_feature_importance.png
│   ├── plot8_elbow.png
│   ├── plot9_pca_clusters.png
│   ├── plot10_dendrogram.png
│   ├── plot11_ann_loss.png
│   └── plot12_word_freq.png
│
├── phase1_data_engine.py
├── phase2_eda.py
├── phase3_ml_models.py
├── phase4_clusters.py
├── phase5_nlp_cv.py
├── phase6_app.py
└── README.md
```

---

## 🧩 Project Pipeline

```text
Raw / Mock Business Data
          │
          ▼
┌─────────────────────────┐
│ Phase 1                 │
│ Data Cleaning &         │
│ Feature Engineering    │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ Phase 2                 │
│ Exploratory Data        │
│ Analysis                 │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ Phase 3                 │
│ Machine Learning        │
│ Models & Evaluation     │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ Phase 4                 │
│ Clustering + PCA + ANN  │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ Phase 5                 │
│ NLP + Sentiment + CNN   │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ Phase 6                 │
│ Streamlit Application   │
└─────────────────────────┘
```

---

## 🛠️ Technologies Used

### Programming

* Python

### Data Analysis

* Pandas
* NumPy

### Visualization

* Matplotlib
* Seaborn

### Machine Learning

* Scikit-learn
* Random Forest
* Logistic Regression
* Decision Tree
* SVM
* KNN
* Naive Bayes

### Unsupervised Learning

* K-Means
* Hierarchical Clustering
* PCA

### NLP

* TF-IDF
* Text preprocessing
* Sentiment analysis

### Deep Learning

* TensorFlow / Keras
* CNN concepts
* Artificial Neural Networks

### Deployment / UI

* Streamlit

---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/YOUR-USERNAME/ShopSurvival.git
cd ShopSurvival
```

Create a virtual environment:

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

Install the required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy joblib streamlit
```

For the optional CNN component:

```bash
pip install tensorflow
```

---

## ▶️ Run the Streamlit Application

The repository already contains processed data inside the `data/` directory, so the final application can be launched directly.

```bash
streamlit run phase6_app.py
```

Open the local URL displayed in your terminal:

```text
http://localhost:8501
```

---

## 🔄 Run the Complete Pipeline

If you want to reproduce the project from the beginning:

```bash
python phase1_data_engine.py
python phase2_eda.py
python phase3_ml_models.py
python phase4_clusters.py
python phase5_nlp_cv.py
streamlit run phase6_app.py
```

> Some phases can operate using generated/mock data when the original dataset is unavailable.

---

## 📊 Dataset

The project is designed around business data containing attributes such as:

* Business name
* City
* Category
* Rating
* Review count
* Competition-related features
* Survival status

The original project documentation references the **Yelp Business Dataset** as the primary external data source.

Dataset source:

**Yelp Dataset — Kaggle**

https://www.kaggle.com/datasets/yelp-dataset/yelp-dataset

---

## 📈 Generated Outputs

The project generates analytical outputs including:

* Survival analysis plots
* Rating distributions
* Correlation heatmaps
* Competition analysis
* Pair plots
* Confusion matrices
* Feature importance
* Elbow curves
* PCA cluster visualization
* Hierarchical clustering dendrogram
* ANN training-loss visualization
* Word-frequency visualization

---

## 💡 What This Project Demonstrates

This project demonstrates an end-to-end Data Science workflow:

```text
Data Collection
      ↓
Data Cleaning
      ↓
Exploratory Data Analysis
      ↓
Statistical Analysis
      ↓
Feature Engineering
      ↓
Machine Learning
      ↓
Model Evaluation
      ↓
Clustering
      ↓
Dimensionality Reduction
      ↓
NLP / Sentiment Analysis
      ↓
Interactive Application
```

It combines multiple areas of Data Science into a single practical project rather than focusing on only one machine-learning algorithm.

---

## ⚠️ Disclaimer

ShopSurvival is an educational Data Science project.

The survival probabilities produced by the application are model-based estimates derived from the available dataset and selected features. They should not be considered financial, business, or investment advice.

---

## 👨‍💻 Author

**Pavan Sai**

B.Tech — Computer Science and Technology

Interested in:

* Data Science
* Machine Learning
* Data Analytics
* Artificial Intelligence

---

## ⭐ If You Find This Project Useful

If this project helped you understand Data Science, Machine Learning, or Streamlit development, consider giving the repository a ⭐.
