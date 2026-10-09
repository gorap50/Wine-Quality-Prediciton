# Wine Quality Prediction

Multi-model classification pipeline predicting wine quality scores from physicochemical properties.
Best result: **Random Forest — 85% accuracy, 0.88 ROC-AUC** on UCI white wine dataset.

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)

---

## Results

| Model | Accuracy | ROC-AUC |
|---|---|---|
| **Random Forest** | **~0.85** | **~0.88** |
| Decision Tree | ~0.81 | — |
| Logistic Regression | ~0.79 | ~0.84 |
| KNN | ~0.78 | — |

Validated with 5-fold cross-validation.

---

## Pipeline

1. **EDA** — distribution analysis, correlation matrix, outlier detection (IQR)
2. **Preprocessing** — MinMaxScaler + StandardScaler, feature selection (threshold > 0.8 correlation dropped)
3. **Class balancing** — SMOTE applied to handle imbalanced quality scores
4. **Modeling** — Random Forest, Decision Tree, KNN, Logistic Regression
5. **Evaluation** — accuracy, confusion matrix, ROC-AUC curve

---

## Dataset

[UCI Wine Quality Dataset](https://archive.ics.uci.edu/ml/datasets/wine+quality) — 11 physicochemical features, quality score 0–10 (white wine).

---

## Run

```bash
git clone https://github.com/gorap50/wine-quality-prediction.git
cd wine-quality-prediction
pip install -r requirements.txt
jupyter notebook wine_quality.ipynb
```
