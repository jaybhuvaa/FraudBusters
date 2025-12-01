# IEEE-CIS Fraud Detection System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.5%2B-green.svg)](https://xgboost.readthedocs.io/)

A comprehensive machine learning pipeline for detecting fraudulent transactions using the IEEE-CIS Fraud Detection dataset from Kaggle.

---

## 👥 Team Members

| Name | Role |
|------|------|
| **Prajakta Avachat** | Data Collection & Preparation |
| **Jay Bhuva** | Model Building & Evaluation |

*Both members collaborated on research, documentation, and presentation.*

**Course:** DS 5110 - Essentials of Data Science  
**Institution:** Northeastern University

## 🎯 Project Overview

Financial fraud is a growing concern in today's digital world, causing billions of dollars in losses annually. This project builds a robust fraud detection system that:

- **Preprocesses** large-scale transaction data with advanced feature engineering
- **Handles class imbalance** using SMOTE (Synthetic Minority Over-sampling Technique)
- **Compares multiple ML models** with focus on recall and precision metrics
- **Stores results in SQL database** with comprehensive analytical reports

### Why Fraud Detection Matters

| Metric | Importance |
|--------|------------|
| **Recall** | Catching frauds prevents financial losses |
| **Precision** | Reducing false alarms improves customer experience |
| **Speed** | Real-time detection is critical |

---

## 📊 Dataset

**Source:** [IEEE-CIS Fraud Detection - Kaggle Competition](https://www.kaggle.com/competitions/ieee-fraud-detection)

**Provider:** Vesta Corporation (real-world e-commerce transactions)

| File | Description | Shape |
|------|-------------|-------|
| `train_transaction.csv` | Transaction data | 590,540 × 394 |
| `train_identity.csv` | Identity/device data | 144,233 × 41 |
| **Merged Dataset** | Combined features | 590,540 × 434 |

### Key Characteristics

- **Class Imbalance:** Only 3.5% fraudulent transactions
- **Missing Values:** Extensive (0.1% to 99% across features)
- **Feature Types:** Numeric (403) + Categorical (31)
- **Anonymized Features:** V1-V339, C1-C14, D1-D15

---

## 📁 Project Structure

```
fraud-detection/
│
├── 📓 Notebooks/
│   ├── EDA.ipynb                      # Exploratory Data Analysis
│   ├── Preprocessing_Complete.ipynb   # Data preprocessing pipeline
│   ├── Model_Building.ipynb           # Model training & evaluation
│   └── SQL_Database_Reports.ipynb     # Database & SQL analytics
│
├── 📂 outputs/
│   ├── fraud_detection.db             # SQLite database
│   ├── X_train_balanced.csv           # Processed training data
│   ├── X_test.csv                     # Test data
│   ├── y_train_balanced.csv           # Training labels
│   ├── y_test.csv                     # Test labels
│   ├── model_comparison.csv           # Model metrics comparison
│   ├── model_predictions.csv          # All model predictions
│   ├── model_performance_summary.csv  # Performance summary
│   ├── feature_importance_*.csv       # Feature importance scores
│   ├── sql_report*.csv                # SQL report exports
│   └── *.png                          # Visualizations
│
├── 📂 models/
│   ├── logistic_regression.pkl        # Trained LR model
│   ├── random_forest.pkl              # Trained RF model
│   ├── xgboost.pkl                    # Trained XGB model
│   └── lightgbm.pkl                   # Trained LGB model
│
├── 📂 data/                           # Raw data (not included)
│   ├── train_transaction.csv
│   └── train_identity.csv
│
└── README.md                          # This file
```

---

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- Jupyter Notebook or JupyterLab

### Required Libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
pip install xgboost lightgbm imbalanced-learn
pip install joblib
```

Or install all at once:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm imbalanced-learn joblib
```

### Dataset Download

1. Go to [Kaggle IEEE-CIS Competition](https://www.kaggle.com/competitions/ieee-fraud-detection/data)
2. Download `train_transaction.csv` and `train_identity.csv`
3. Place files in the project root or `data/` folder

---

## 🚀 Usage

### Execution Order

Run the notebooks in this sequence:

```
1. EDA.ipynb                        # Understand the data
        ↓
2. Preprocessing_Complete.ipynb     # Clean & prepare data
        ↓
3. Model_Building.ipynb             # Train & evaluate models
        ↓
4. SQL_Database_Reports.ipynb       # Generate database & reports
```

### Quick Start

```python
# Load preprocessed data
import pandas as pd

X_train = pd.read_csv('outputs/X_train_balanced.csv')
X_test = pd.read_csv('outputs/X_test.csv')
y_train = pd.read_csv('outputs/y_train_balanced.csv')
y_test = pd.read_csv('outputs/y_test.csv')

# Load trained model
import joblib
model = joblib.load('models/xgboost.pkl')

# Make predictions
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]
```

---

## 🔬 Methodology

### 1. Data Preprocessing

| Step | Technique | Details |
|------|-----------|---------|
| **Merging** | Left Join | Combined transaction + identity on TransactionID |
| **Missing Values** | Threshold removal | Dropped features with >80% missing |
| **Numeric Imputation** | Median | Robust to outliers |
| **Categorical Imputation** | Mode/'Unknown' | Preserved information |
| **Encoding** | Label + One-Hot | Based on cardinality threshold |
| **Scaling** | StandardScaler | For linear models |

### 2. Feature Engineering

Created 11 new features:

| Category | Features |
|----------|----------|
| **Time-based** | `Transaction_hour`, `Transaction_day_of_week`, `is_weekend`, `is_night` |
| **Amount-based** | `TransactionAmt_log`, `TransactionAmt_decimal`, `TransactionAmt_is_round` |
| **Email-based** | `P_email_suffix`, `R_email_suffix`, `email_match` |
| **Interactions** | `card1_card2`, `addr1_addr2` |

### 3. Class Imbalance Handling

```
Before SMOTE:
  Non-Fraud: 455,902 (96.5%)
  Fraud: 16,530 (3.5%)
  Ratio: 27.6:1

After SMOTE:
  Non-Fraud: 455,902 (66.7%)
  Fraud: 227,951 (33.3%)
  Ratio: 2:1
```

### 4. Models Trained

| Model | Description |
|-------|-------------|
| **Logistic Regression** | Linear baseline with L2 regularization |
| **Random Forest** | Ensemble of 100 decision trees |
| **XGBoost** | Gradient boosting with regularization |
| **LightGBM** | Fast gradient boosting framework |

---

## 📈 Results

### Model Comparison

| Model                | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|----------------------|----------|-----------|--------|----------|---------|
| Logistic Regression  | 0.9470   | 0.3222    | 0.4662 | 0.3811   | 0.8332  |
| Random Forest        | 0.9730   | 0.6613    | 0.4687 | 0.5486   | 0.8976  |
| **XGBoost**          | **0.9748** | **0.7195** | **0.4587** | **0.5603** | **0.9062** |
| LightGBM             | 0.9738   | 0.6918    | 0.4529 | 0.5474   | 0.9042  |


*Note: Actual results may vary based on random seed and data splits.*

### Best Model: XGBoost

- **Highest AUC-ROC:** 0.9062 (excellent discrimination)
- **Best Recall:** 0.4587 (catches 46% of frauds)
- **Balanced F1-Score:** 0.5603

### Top 10 Important Features

1. TransactionAmt
2. card1
3. card2
4. addr1
5. D15
6. D10
7. D4
8. C13
9. C1
10. Transaction_hour

---

## 🗄️ SQL Database & Reports

### Database Schema (ER Diagram)

```
+------------------+         +------------------+
|    CUSTOMERS     |         |   TRANSACTIONS   |
+------------------+         +------------------+
| PK customer_id   |◄───────►| PK transaction_id|
|    card1         |   1:N   | FK customer_id   |
|    card2         |         |    amount        |
|    card4         |         |    product_cd    |
|    card6         |         |    is_fraud      |
|    addr1, addr2  |         |    trans_hour    |
+------------------+         +--------┬---------+
                                      │ 1:1
                                      ▼
                             +------------------+
                             | MODEL_PREDICTIONS|
                             +------------------+
                             | FK transaction_id|
                             |    y_true        |
                             |    lr_prediction |
                             |    rf_prediction |
                             |    xgb_prediction|
                             |    lgb_prediction|
                             +------------------+

+------------------+
| MODEL_PERFORMANCE|
+------------------+
| PK model_id      |
|    model_name    |
|    accuracy      |
|    precision     |
|    recall        |
|    f1_score      |
|    auc_roc       |
+------------------+
```

### 10 Advanced SQL Reports

| # | Report | Key Insight |
|---|--------|-------------|
| 1 | Fraud by Product Type | Product 'C' has highest fraud rate (11.69%) |
| 2 | Fraud by Hour | Hours 6-9 AM have highest fraud rates (7-10%) |
| 3 | Top Customers | Top 20 customers by transaction volume |
| 4 | Amount Comparison | Fraudulent txns avg $149 vs $134 normal |
| 5 | Daily Trend | Fraud patterns over time |
| 6 | Model Performance | All model metrics in SQL |
| 7 | Model Flags | Transactions flagged by each model |
| 8 | Customer Segments | High-risk card type combinations |
| 9 | Precision/Recall | Confusion matrix per model |
| 10 | Device Fraud | Fraud rate by device type |

---

## 🔍 Key Findings

### Fraud Patterns Discovered

1. **Time Matters:** Early morning (6-9 AM) has 2-3x higher fraud rate
2. **Product Risk:** Digital products (ProductCD='C') are most targeted
3. **Amount Signals:** Fraudulent transactions are slightly higher ($149 vs $134)
4. **Device Patterns:** Unknown devices show higher fraud likelihood

### Business Recommendations

1. **Enhanced monitoring** during early morning hours
2. **Additional verification** for digital product purchases
3. **Device fingerprinting** to reduce unknown device fraud
4. **Real-time scoring** using XGBoost model for best results

---

## 📚 References

1. **Dataset:** [IEEE-CIS Fraud Detection - Kaggle](https://www.kaggle.com/competitions/ieee-fraud-detection)
2. **SMOTE:** Chawla, N.V., et al. (2002). "SMOTE: Synthetic Minority Over-sampling Technique"
3. **XGBoost:** Chen, T., & Guestrin, C. (2016). "XGBoost: A Scalable Tree Boosting System"
4. **LightGBM:** Ke, G., et al. (2017). "LightGBM: A Highly Efficient Gradient Boosting Decision Tree"

---

## 📄 License

This project is for educational purposes as part of DS 5110 coursework at Northeastern University.

---

## 🙏 Acknowledgments

- **IEEE Computational Intelligence Society** for organizing the competition
- **Kaggle** for hosting the dataset
- **Northeastern University** for course guidance

---

<p align="center">
  <b>DS 5110 - Essentials of Data Science</b><br>
  Northeastern University | Fall 2025
</p>
