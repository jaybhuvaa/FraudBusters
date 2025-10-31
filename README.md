# 💳 Fraud Detection in Finance (IEEE-CIS Dataset)

### Team Members
- **Prajakta Avachat** – Data Preprocessing & Visualization  
- **Jay Bhuva** – Machine Learning & Model Evaluation  

---

## 🧭 Project Overview

Financial fraud is one of the most critical challenges in the digital era, where thousands of online transactions happen every second.  
This project — **Fraud Detection in Finance** — leverages machine learning models to detect and classify potentially fraudulent transactions based on the **IEEE-CIS Fraud Detection Dataset**.

Our goal is to design, train, and evaluate predictive models that can effectively identify fraud using engineered data patterns and transaction metadata.

---

## 🎯 Goals and Expected Outcomes

### **Goals**
- Build a machine learning pipeline capable of detecting fraudulent transactions.
- Conduct data preprocessing, feature engineering, and exploratory data analysis (EDA).
- Evaluate and compare multiple models for optimal fraud classification performance.

### **Expected Outcomes**
- Trained ML models with metrics including **Accuracy**, **Precision**, **Recall**, **F1-score**, and **ROC-AUC**.
- Visualization reports showing the impact of key variables.
- Reproducible implementation hosted on GitHub, with detailed documentation.

---

## 📦 Dataset Information

**Source:** [IEEE-CIS Fraud Detection Dataset](https://www.kaggle.com/competitions/ieee-fraud-detection/data?select=test_transaction.csv)

**Description:**  
The dataset contains **transaction** and **identity** information for millions of online payments.  
The goal is to predict whether each transaction is fraudulent (`isFraud = 1`) or legitimate (`isFraud = 0`).

**Files Used:**
| File | Description |
|------|--------------|
| `train_transaction.csv` | Transaction-level features and fraud labels |
| `train_identity.csv` | Device, browser, and identity-related metadata |
| `test_transaction.csv` | Unlabeled transaction data for prediction |
| `test_identity.csv` | Unlabeled identity information |

**Merged Dataset:** Joined using `TransactionID`.

---

## ⚙️ Project Scope

### **In Scope**
- Data cleaning, EDA, and feature selection using Python.
- Model training using algorithms such as:
  - Logistic Regression  
  - Random Forest  
  - XGBoost
- Handling data imbalance via SMOTE / undersampling.
- Model evaluation with ROC-AUC and F1-score metrics.

### **Out of Scope**
- Integration with live banking APIs.  
- Deployment of real-time fraud detection systems.

---

## 🗓️ Key Deliverables and Milestones

| Phase | Deliverable | Target Week |
|-------|--------------|--------------|
| **Phase 1** | Dataset exploration & setup | Week 2 |
| **Phase 2** | Data preprocessing & feature engineering | Week 3 |
| **Phase 3** | Model training & evaluation | Week 5 |
| **Phase 4** | Documentation & final report | Week 6 |

---

## 👥 Team Roles and Responsibilities

| Team Member | Core Skills | Responsibilities |
|--------------|-------------|------------------|
| **Prajakta Avachat** | Python, Data Preprocessing, Visualization | Data cleaning, feature analysis, visualization |
| **Jay Bhuva** | Machine Learning, Model Evaluation, Documentation | Model training, evaluation, and reporting |

---

## 🧰 Tools and Technologies

| Category | Tools |
|-----------|-------|
| **Language** | Python 3.x |
| **Libraries** | `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `xgboost` |
| **Environment** | Jupyter Notebook / Google Colab |
| **Version Control** | GitHub (branching strategy: `main` & `dev`) |
| **Documentation** | Overleaf (reports), Excel (progress tracking) |

---

## 🧪 Development Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/jaybhuvaa/FraudBusters.git
   cd FraudBusters
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the Dataset**
   - Sign in to [Kaggle IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection/data)
   - Download `train_transaction.csv` and `train_identity.csv`
   - Place them in a `/data` folder in the repo.

4. **Run the EDA Script**
   ```bash
   python scripts/explore_train_data.py
   ```
   This generates:
   - Missing value summary  
   - Numeric & categorical analysis  
   - Fraud distribution visualizations  

5. **Train Models**
   ```bash
   python scripts/train_models.py
   ```

---

## 📊 Exploratory Data Analysis (EDA)

Key aspects explored:
- Data volume and feature overview  
- Missing values & sparsity patterns  
- Feature distribution visualization  
- Correlation heatmaps with `isFraud`  
- Categorical frequency plots  
- Outlier analysis and transformation (`log(TransactionAmt)`)

Outputs:
- `numeric_summary.csv`
- `categorical_summary.csv`
- `missing_summary.csv`

---

## 🤖 Model Evaluation Metrics

| Metric | Description |
|---------|-------------|
| **Accuracy** | Overall correctness of predictions |
| **Precision** | Proportion of correctly predicted frauds among all fraud predictions |
| **Recall** | Ability to find all actual frauds |
| **F1-score** | Balance between precision and recall |
| **ROC-AUC** | Probability that model ranks a fraud higher than a non-fraud |

---

## 🧩 Results Summary *(To Be Updated After Final Model Run)*

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|--------|-----------|-----------|--------|----|----------|
| Logistic Regression | TBD | TBD | TBD | TBD | TBD |
| Random Forest | TBD | TBD | TBD | TBD | TBD |
| XGBoost | TBD | TBD | TBD | TBD | TBD |

---


---

## 📈 Future Improvements

- Implement LightGBM & CatBoost for performance tuning.
- Apply deep learning (autoencoders) for anomaly detection.
- Deploy a basic Flask dashboard for result visualization.

---

## 📚 References

- [IEEE-CIS Fraud Detection on Kaggle](https://www.kaggle.com/competitions/ieee-fraud-detection)
- [Top 5 Kaggle Solutions](https://medium.com/data-science/ieee-cis-fraud-detection-top-5-solution-5488fc66e95f)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)

---

## 🏁 Acknowledgment

This project was developed as part of the **End of Degree (EOD) Submission** for academic purposes, focusing on the application of data science and machine learning in financial fraud detection.
