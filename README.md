# solar-panel-efficiency-ml
# Solar Panel Efficiency Prediction using Machine Learning

This project implements a robust and scalable machine learning workflow to predict solar panel efficiency using sensor readings and error code data. Designed with modularity, reproducibility, and cloud execution efficiency (e.g., Google Colab) in mind, the pipeline leverages preprocessing pipelines, hyperparameter tuning, and custom evaluation metrics to optimize performance.

---

## Objective

Predict the efficiency of solar panels using historical sensor and operational data to enable predictive maintenance and maximize energy yield.

---

## Files

- train.csv – Training data with features and target.
- test.csv – Test data without target (for final prediction).
- submission_final.csv – Output file with model predictions.
- notebook.ipynb – Main Jupyter notebook containing the full workflow.

---

## ML Workflow Overview

### 1. Data Loading

- Loads `train.csv` and `test.csv` into Pandas DataFrames.
- Assumes availability of these files in local or cloud paths (e.g., Google Drive or GitHub).

---

### 2. Feature Engineering

- Numerical Conversion: Converts voltage, current, temperature, and humidity to `float`, coercing errors to NaN.
- Derived Feature: Power = voltage × current, capturing key physical insights.
- Categorical Handling: Fills missing `error_code` with "Normal" and casts to string.

These steps ensure data consistency and create features that capture domain-specific relationships.

---

### 3. Target and Feature Setup

- Target Variable: `efficiency`
- Feature Lists:
  - numerical_features: Sensor values, power, etc.
  - categorical_features: Error codes
- Cleaning: Drops rows with missing target or critical features in training.

---

### 4. Preprocessing Pipeline

Implemented using scikit-learn's `Pipeline` and `ColumnTransformer`:

- Numerical Pipeline:
  - `SimpleImputer(strategy='most_frequent')`
  - `StandardScaler()`

- Categorical Pipeline:
  - `SimpleImputer(strategy='most_frequent')`
  - `OneHotEncoder(handle_unknown='ignore')`

Ensures preprocessing steps are applied consistently and avoids data leakage.

---

### 5. Custom Evaluation Metric

A custom score is defined for interpretability:

Score = 100 × (1 - sqrt(MSE))

- Rewards low RMSE with higher scores.
- Scales performance to a 0-100 range.

---

### 6. Model Selection and Tuning

- Model Used: `RandomForestRegressor` – handles non-linear patterns and is robust to overfitting.
- Tuning: `RandomizedSearchCV` with:
  - n_estimators, max_depth, max_features
  - min_samples_split, min_samples_leaf, bootstrap
- Validation: 3-fold cross-validation for efficiency in Google Colab.

---

### 7. Cross-Validation

- Method: `KFold(n_splits=5)`
- Metrics:
  - MSE
  - RMSE
  - R² Score
  - Custom Score

Prints fold-wise metrics and averages for model stability and generalization insight.

---

### 8. Final Model Training and Prediction

- Retrain: Best pipeline is trained on full training data.
- Predict: Applies preprocessing and model to `test.csv`.

---

### 9. Submission File

- Combines predictions with test set IDs.
- Asserts shape to catch bugs early.
- Saves results as `submission_final.csv`.

---

## Design Highlights

- Modular Pipelines for reproducibility and clarity
- Custom Scoring for interpretability
- Colab Optimized: Efficient parameter grid and cross-validation strategy
- Feature Engineering: Power feature leverages physics for deeper insight
- Robust Missing Data Handling using `SimpleImputer`

---

## Getting Started

```bash
# Clone the repo
git clone https://github.com/archashikamal/solar-panel-efficiency-ml.git
cd solar-panel-efficiency-ml

# Open in Colab or Jupyter



