# 🍷 Wine Quality Prediction using Machine Learning

This project uses machine learning techniques to predict the quality of wine based on its physicochemical properties. Multiple classifiers were tested and compared to find the most accurate model. The dataset used is derived from white wine samples and includes features such as acidity, sugar, pH, alcohol content, and more.

---

## 📌 Project Objectives

- Analyze and preprocess the wine dataset.
- Detect and handle missing values, duplicates, and outliers.
- Normalize and scale features using StandardScaler and MinMaxScaler.
- Balance the dataset using SMOTE to handle class imbalance.
- Train and evaluate multiple classification models:
  - K-Nearest Neighbors (KNN)
  - Decision Tree
  - Random Forest
  - Logistic Regression
- Compare model performance using accuracy, confusion matrix, and ROC-AUC.

---

## 🧠 Technologies & Libraries Used

- Python 🐍
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- imbalanced-learn (SMOTE)

---

## 📂 Dataset

- **Name:** Wine Quality Dataset (White Wine)
- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/wine+quality)
- **Features:** 11 physicochemical attributes (e.g., acidity, sugar, pH)
- **Target:** Wine quality score (0–10)

---

## 🔍 Exploratory Data Analysis (EDA)

- Checked for missing values and duplicate rows.
- Visualized distributions using histograms and boxplots.
- Correlation matrix and scatter plots to detect relationships.
- Removed highly correlated features (threshold > 0.8).

---

## 🧹 Preprocessing Steps

- Normalization using MinMaxScaler.
- Standardization using StandardScaler.
- Outlier detection via IQR method.
- Applied SMOTE to balance target classes.

---

## 🤖 Model Training & Evaluation

| Model             | Accuracy | ROC AUC |
|------------------|----------|---------|
| Random Forest     | ~0.85    | ~0.88   |
| Decision Tree     | ~0.81    | -       |
| KNN               | ~0.78    | -       |
| Logistic Regression | ~0.79  | ~0.84   |

- Accuracy evaluated using `accuracy_score`
- Model robustness checked using 5-fold `cross_val_score`

---

## 📈 Visualizations

- Boxplots for feature distribution
- Heatmaps for feature correlation
- Countplot for class imbalance
- ROC Curve for model comparison

---

## 🚀 How to Run

1. Clone the repository  
   ```bash
   git clone https://github.com/yourusername/wine-quality-prediction.git
   cd wine-quality-prediction


