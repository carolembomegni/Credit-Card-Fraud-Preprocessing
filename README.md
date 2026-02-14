#  Credit Card Fraud Detection – Data Preprocessing & ML Pipeline

##  Project Overview

This project focuses on applying advanced data preprocessing techniques and building a complete machine learning pipeline for credit card fraud detection.

The dataset contains European credit card transactions made in September 2013 over a two-day period.

- Total transactions: **284,807**
- Fraud cases: **492**
- Fraud rate: **0.172%**

This is a highly imbalanced classification problem.

---

##  Objectives

- Perform structured data preprocessing
- Handle class imbalance
- Apply scaling and transformation techniques
- Generate polynomial features
- Build a complete ML pipeline using `ColumnTransformer`
- Evaluate the model using appropriate metrics for imbalanced data (AUPRC)

---

##  Dataset Information

The dataset contains only numerical features:

- **V1–V28**: Principal Components obtained using PCA (for confidentiality)
- **Time**: Seconds elapsed between transactions
- **Amount**: Transaction amount
- **Class**: Target variable (0 = normal, 1 = fraud)

Dataset source:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

---

##  Methodology

### 1️ Data Exploration
- Verified dataset dimensions
- Checked for missing values (none found)
- Analyzed class imbalance
- Identified strong skewness in `Amount`

---

### 2️ Imputation
Although no missing values were found, `SimpleImputer` was included in the pipeline to ensure robustness in real-world scenarios.

---

### 3️ Encoding
No categorical variables were present in the dataset; therefore, no encoding was required.

---

### 4️ Discretization
`KBinsDiscretizer` (strategy = quantile) was applied to `Amount` to analyze distribution behavior before and after discretization.

---

### 5️ Scaling Comparison

Three scaling methods were compared:

- MinMaxScaler
- StandardScaler
- RobustScaler

Due to strong outliers in `Amount`, **RobustScaler** was selected.

---

### 6️ Power Transformation

`PowerTransformer (Yeo-Johnson)` was applied to reduce skewness in `Amount`.

Result:
- Improved symmetry
- Reduced skewness
- Better suitability for linear models

---

### 7️ Polynomial Features

PolynomialFeatures (degree = 2) was applied to `Amount_PT`.

Generated features:
- Amount_PT
- Amount_PT²

This allows the model to capture potential non-linear relationships.

---

### 8 Final Pipeline

The final pipeline includes:

- Imputation (SimpleImputer)
- Scaling (RobustScaler)
- Logistic Regression (class_weight="balanced")

Built using:
- `ColumnTransformer`
- `make_pipeline`

---

##  Model Evaluation

### Confusion Matrix Results

- True Positives (Fraud detected): 90
- False Negatives (Missed fraud): 8
- True Negatives: 55,481
- False Positives: 1,383

Recall (Fraud): **≈ 92%**

---

### Threshold Optimization

Default threshold (0.5) produced high recall but low precision.

By adjusting the decision threshold (e.g., 0.7):

- Improved precision
- Maintained strong recall
- Reduced false positives

This demonstrates understanding of precision-recall trade-offs in imbalanced problems.

---

### AUPRC Score

**Average Precision Score (AUPRC): ≈ 0.71**

Given the extreme imbalance (baseline ≈ 0.0017), this result demonstrates strong discriminative power.

---

##  Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

---

## Key Takeaways

- Preprocessing is critical in imbalanced classification.
- Robust scaling outperforms MinMax and Standard in skewed distributions.
- Power transformations improve feature distribution.
- Threshold tuning significantly impacts business outcomes.
- AUPRC is more appropriate than accuracy for imbalanced datasets.

---

## Conclusion

This project demonstrates a structured preprocessing workflow and a complete machine learning pipeline tailored for highly imbalanced classification problems.

The model successfully detects the majority of fraud cases while allowing threshold adjustment based on business constraints.

---

 Author: Carole Mbomegni Nana  
 Data Science Student | Applied Machine Learning | Financial Analytics
