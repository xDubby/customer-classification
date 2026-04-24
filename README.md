# Customer Credit Risk Classification

A supervised machine learning project to classify bank customers as **Good** or **Bad** credit risk based on financial and demographic indicators.

The model is deployed as an interactive Streamlit dashboard where users can input customer data and receive a real-time risk prediction with probability breakdown.

---

## Project Overview

**Dataset:** German Credit Data — UCI Machine Learning Repository (1000 records, 20 features)  
**Task:** Binary classification (Good credit = 0, Bad credit = 1)  
**Stack:** Python, Scikit-learn, XGBoost, SMOTE, Streamlit  

The dataset contains real-world financial data including checking account status, credit history, savings, employment duration, loan amount and purpose.

---

## Results

Two models are included, each optimized for a different objective:

| Model | AUC (test) | AUC (cross-val) | Recall Bad | Use case |
|---|---|---|---|---|
| Random Forest | 0.795 | 0.781 | 0.55 | Maximize overall predictive power |
| XGBoost + SMOTE | 0.759 | 0.773 | 0.58 | Minimize missed bad customers |

In a banking context, failing to identify a bad customer (false negative) carries a higher cost than incorrectly flagging a good one (false positive). For this reason, **XGBoost with SMOTE** is the recommended model for production use, as it achieves a higher recall on the Bad class by addressing the class imbalance (70% Good / 30% Bad).

---

## Key Findings from EDA

- `checking` (account balance status) is the most predictive feature in both models
- Customers with savings below 100 DM show the highest proportion of bad risk
- Longer loan durations are associated with higher default probability
- `credit_history` shows a counterintuitive pattern: "critical account" clients are predominantly good risk, likely due to stricter screening

---

## Project Structure

```
customer-classification/
├── data/                   # Raw dataset
├── notebooks/
│   └── 01_EDA_clean.ipynb  # Exploratory analysis, feature engineering, model training
├── src/
│   ├── model_xgb.pkl       # XGBoost model (SMOTE, recall-optimized)
│   ├── model_rf.pkl        # Random Forest model (AUC-optimized)
│   └── scaler.pkl          # StandardScaler for numeric features
├── app.py                  # Streamlit dashboard
├── requirements.txt        # Python dependencies
└── .gitignore
```

---

## How to Run

**1. Clone the repository**
```bash
git clone https://github.com/xDubby/customer-classification.git
cd customer-classification
```

**2. Create and activate a virtual environment**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Launch the dashboard**
```bash
streamlit run app.py
```

The app will open automatically at `http://localhost:8501`.

---

## Dashboard

The Streamlit app allows users to:

- Input customer financial data via sidebar controls
- Choose between the two trained models
- Get an instant Good/Bad classification with probability scores
- View a visual breakdown of risk probability

---

## Methodology

**Feature Engineering**
- Label encoding of 13 categorical variables
- StandardScaler normalization of 7 numeric features

**Model Selection**
- Logistic Regression, Random Forest and XGBoost trained and evaluated
- 5-fold cross-validation with ROC-AUC scoring
- SMOTE applied exclusively on the training set to avoid data leakage

**Evaluation**
- ROC-AUC as primary metric
- Precision, Recall and F1-score per class
- Comparison with and without SMOTE balancing

---

## Dataset Reference

Hofmann, H. (1994). Statlog (German Credit Data). UCI Machine Learning Repository.  
[https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data)

---

## Author

Federico D'Ubaldi  
[LinkedIn](https://www.linkedin.com/in/federicodubaldi) | [federicodubaldi.it](https://federicodubaldi.it)
