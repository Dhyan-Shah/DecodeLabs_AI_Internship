# Project 2 — Data Classification Using AI

**DecodeLabs Industrial Training Kit | Batch 2026**

A supervised machine learning project that builds a K-Nearest Neighbors (KNN) classifier on the Iris dataset — covering the full pipeline from raw data to model evaluation.

---

## Overview

This project demonstrates the fundamental ML pipeline:

- Loading and exploring a real dataset
- Preprocessing with feature scaling
- Splitting data into training and testing sets
- Finding the optimal hyperparameter (K) using the Elbow method
- Training a KNN classification model
- Evaluating with accuracy, F1 score, and confusion matrix
- Predicting on new unseen data

---

## Project Structure

```
project2/
│
├── project2_iris_knn.py      # Main script (full pipeline)
├── project2_results.png      # Output visualizations
└── README.md                 # This file
```

---

## Algorithm Used

**K-Nearest Neighbors (KNN)**

KNN classifies a new data point by looking at its K closest neighbors in the training data and taking a majority vote. It follows the *Proximity Principle* — similar things exist in close proximity.

| K too low (K=1) | K optimal (elbow) | K too high (K=100) |
|---|---|---|
| Overfitting | Balanced | Underfitting |
| Sensitive to noise | Best generalization | Too generic |

---

## Dataset — The Iris Benchmark

| Property | Value |
|---|---|
| Source | `sklearn.datasets.load_iris()` |
| Samples | 150 (balanced) |
| Classes | 3 (Setosa, Versicolor, Virginica) |
| Features | 4 (sepal length, sepal width, petal length, petal width) |

---

## IPO Framework

```
INPUT               PROCESS                 OUTPUT
─────────────       ───────────────────     ──────────────────
Iris dataset   →    Train-Test Split   →    Confusion Matrix
Feature Scaling     KNN Algorithm           F1 Score
```

---

## Setup & Installation

### Prerequisites

- Python 3.7+
- pip

### Install dependencies

```bash
pip install scikit-learn pandas matplotlib seaborn numpy
```

---

## How to Run

```bash
python project2_iris_knn.py
```

### What happens when you run it

1. Dataset is loaded and printed to console
2. Features are scaled using `StandardScaler`
3. Data is split 80% training / 20% testing
4. Elbow method runs K from 1 to 20 and picks the best K
5. KNN model is trained and predictions are made
6. Accuracy, F1 score, and classification report are printed
7. Three charts are saved as `project2_results.png`
8. Three new flower samples are classified and printed

---

## Pipeline Breakdown

### Step 1 — Load & Understand Data
```python
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
```
Loads the Iris dataset and converts it to a readable DataFrame for inspection.

### Step 2 — Feature Scaling
```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```
Normalizes all features to mean=0, variance=1 so no single feature dominates distance calculations in KNN.

### Step 3 — Train-Test Split
```python
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, shuffle=True
)
```
Splits data into 120 training samples and 30 testing samples. Data is shuffled first to remove order bias.

### Step 4 — Find Optimal K (Elbow Method)
```python
error_rates.append(1 - accuracy_score(y_test, preds))
optimal_k = k_range[np.argmin(error_rates)]
```
Trains a temporary KNN for each K value from 1–20 and records the error rate. The K with the lowest error rate is selected. `1 - accuracy` converts accuracy into error rate, making the elbow (dip) easy to spot visually.

### Step 5 — Train the Model
```python
model = KNeighborsClassifier(n_neighbors=optimal_k)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```
Three lines: build the model, memorize the training data, classify the test set.

### Step 6 — Evaluate
```python
accuracy_score(y_test, predictions)   # Overall correctness
f1_score(y_test, predictions, ...)    # Precision-Recall balance
confusion_matrix(y_test, predictions) # Per-class breakdown
```
Multiple metrics are used because accuracy alone can be misleading on imbalanced datasets.

### Step 7 — Visualize
Three charts saved to `project2_results.png`:
- **Elbow curve** — error rate vs K value
- **Confusion matrix heatmap** — predicted vs actual classes
- **Feature scatter plot** — petal length vs petal width colored by species

### Step 8 — Predict New Data
```python
new_scaled = scaler.transform(new_samples)   # Must scale new data too!
new_preds  = model.predict(new_scaled)
```
New samples must pass through the same scaler used during training before prediction.

---

## Sample Output

```
========================================================
  PROJECT 2 — DATA CLASSIFICATION USING AI
  DecodeLabs | Batch 2026
========================================================

📦 Dataset Overview:
   Samples   : 150
   Features  : 4
   Classes   : ['setosa' 'versicolor' 'virginica']

⚖️  Feature Scaling Applied: Mean=0, Variance=1
✂️  Train-Test Split:
   Training samples : 120
   Testing  samples : 30

🔧 Optimal K (Elbow Method): K = 3
✅ Model Trained Successfully!

========================================================
  📈 MODEL EVALUATION RESULTS
========================================================
  Accuracy  : 100.00%
  F1 Score  : 1.0000



---

## Key Concepts Learned

| Concept | Description |
|---|---|
| Supervised Learning | Model learns from labeled examples |
| Feature Scaling | Prevents large-range features from dominating |
| Train-Test Split | Ensures honest evaluation on unseen data |
| Elbow Method | Finds optimal K by minimizing error rate |
| Confusion Matrix | Full breakdown of correct and incorrect predictions |
| F1 Score | Harmonic mean of precision and recall |
| Overfitting | Model too specific to training data (low K) |
| Underfitting | Model too generic (high K) |

---

## Metrics Explained

**Accuracy** — percentage of correct predictions overall.

**Precision** — of all samples predicted as class X, how many actually were X. (Trustworthiness)

**Recall** — of all samples that actually are class X, how many did the model catch. (Sensitivity)

**F1 Score** — harmonic mean of precision and recall. Better than accuracy for class-imbalanced data.

**Confusion Matrix** — a grid showing TP (True Positive), FP (False Positive), FN (False Negative), TN (True Negative) for each class.

---

## Possible Extensions

- Try other algorithms: Decision Tree, SVM, Logistic Regression
- Use cross-validation instead of a single train-test split
- Test on a different dataset (e.g. Wine, Breast Cancer from sklearn)
- Add a GUI to input flower measurements and get live predictions
- Export the trained model using `joblib` for deployment

---

## References

- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Iris Dataset — UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/iris)
- DecodeLabs Industrial Training Kit — Batch 2026

---

## Author

**DecodeLabs Industrial Training Program**
Batch 2026 | Project 2 — Data Classification Using AI