# Abstract  

This project focuses on **fraud detection in financial transactions** using synthetic data.  
The primary objective was to identify potentially fraudulent activities by analyzing transaction patterns, handling imbalanced datasets, and applying machine learning models.  

The workflow involved:  
1. **Data Preprocessing & Visualization** – Cleaning and exploring transaction features such as `Amount`, `Time`, `Location`, and `Device` to detect anomalies and outliers.  
2. **Class Balancing** – Addressing severe class imbalance in the target variable (`IsFraud`) using **undersampling** and **SMOTE** techniques.  
3. **Model Development** – Implementing **Random Forest** and **Gradient Boosting** classifiers to predict fraudulent transactions.  
4. **Evaluation** – Measuring model performance with confusion matrices, precision, recall, and accuracy to assess trade-offs between detecting fraud and minimizing false positives.  

Despite challenges such as extreme class imbalance and overlapping feature distributions, the project demonstrated how **data preprocessing, resampling, and ensemble learning** can contribute toward building an effective fraud detection system. The insights gained highlight the importance of proper data handling and evaluation metrics in **real-world fraud detection applications**.  

# Dataset Overview  

The dataset used in this project is a **synthetic financial transaction dataset** consisting of **50,000 records**. Each record represents a single transaction with features describing its characteristics.  

### Data Summary
- **TransactionID** *(int64)* – Unique identifier for each transaction.  
- **Amount** *(float64)* – Transaction amount in currency units, generated using an exponential distribution to simulate real-world spending patterns.  
- **Time** *(int32)* – Time of the transaction in seconds within a day (0–86,400).  
- **Location** *(object)* – The geographical location where the transaction occurred (e.g., NY, CA, TX, FL).  
- **Device** *(object)* – The device type used for the transaction (e.g., Mobile, Web, ATM).  
- **IsFraud** *(int64)* – Target variable; `0` indicates a genuine transaction, and `1` indicates a fraudulent transaction.  

### Class Distribution
The target variable `IsFraud` is highly **imbalanced**, with approximately **98% genuine transactions** and only **2% fraudulent transactions**. This imbalance makes fraud detection challenging, as models may be biased toward predicting the majority (non-fraudulent) class.  

### Memory Usage
- Total memory used: **~2.1 MB**  
- Data types: Mixture of numerical (`int`, `float`) and categorical (`object`) features.  

This dataset closely resembles real-world financial transaction data where fraud cases are **rare but critical to detect**. Hence, special attention was given to handling class imbalance and ensuring model robustness.

## Descriptive Statistics – Summary  

| Feature        | Mean     | Std Dev   | Min   | 50%   | Max   |
|----------------|----------|-----------|-------|-------|-------|
| TransactionID  | 24,999.5 | 14,433.9  | 0     | 24,999.5 | 49,999 |
| Amount         | 100.3    | 100.8     | 0.00  | 69.1  | 1,021.3 |
| Time           | 43,437.9 | 24,959.4  | 1     | 43,523.5 | 86,397 |
| IsFraud        | 0.0206   | 0.142     | 0     | 0     | 1 |

### 🔑 Key Insights  
- **Amounts**: Avg ≈ **$100**; 75% of transactions are below **$139**; max **$1,021** (possible outlier).  
- **Time**: Spans **1–86,397s (~24h)**, evenly distributed across the day.  
- **Fraud**: Only **~2% fraudulent**, confirming a **highly imbalanced dataset**.  
- **TransactionID**: Just an identifier, not a predictive feature.  

- The dataset contains **no missing values** across all features, so **no imputation is required** at this stage.

## Duplicate Records – Summary  

- **Duplicated Rows Found:** `0`  

### ✅ Key Insight  
- The dataset contains **no duplicate records**, ensuring data consistency and integrity.  

## Location Distribution  

| Location | Count  |
|----------|--------|
| NY       | 12,533 |
| TX       | 12,526 |
| FL       | 12,486 |
| CA       | 12,455 |

### 🌍 Key Insight  
- The dataset is **well-balanced across locations**, with each state contributing nearly equal transactions (~12.5K each).  
- No single location dominates the dataset, reducing the risk of **location bias** in modeling.  

## Device Distribution  

| Device  | Count  |
|---------|--------|
| Mobile  | 16,803 |
| ATM     | 16,703 |
| Web     | 16,494 |

### 📱 Key Insight  
- Transactions are **almost evenly distributed** across devices.  
- **Mobile** is slightly the most used channel, followed by **ATM** and **Web**, but the differences are small.  
- This balance ensures that no device type disproportionately skews the dataset.  


## Transaction Amount (Raw) – Summary  

| Metric | Value   |
|--------|---------|
| Count  | 50,000  |
| Mean   | 100.75  |
| Std    | 100.31  |
| Min    | 0.00    |
| 25%    | 28.75   |
| 50%    | 70.22   |
| 75%    | 140.50  |
| Max    | 1,133.36 |

### 💡 Key Insight  
- The **average transaction amount** is about **$100.75**, but there’s **high variability** (std ≈ 100).  
- 50% of transactions are below **$70.22**, and 75% are below **$140.50**.  
- The **maximum amount $1,133.36** is much larger than the mean → indicates **skewness & potential outliers**.  
- This justifies applying a **log transformation** to stabilize variance and reduce skew.  

## Transaction Time – Summary  

| Metric | Value    |
|--------|----------|
| Count  | 50,000   |
| Mean   | 43,065.94 |
| Std    | 25,035.93 |
| Min    | 6        |
| 25%    | 21,430.50 |
| 50%    | 42,906.50 |
| 75%    | 64,714.75 |
| Max    | 86,394   |

### ⏱️ Key Insight  
- The `Time` feature spans **6 to 86,394 seconds**, almost a full day (~24h = 86,400s).  
- Median ≈ **42,907s (~11.9h)**, meaning half the transactions occur before midday.  
- The distribution appears **uniform across the day**, with no strong concentration in specific time intervals.  
- Could be useful for **time-based fraud detection patterns** (e.g., unusual hours).  

### Distribution of Transactions by Location  

The dataset contains transactions from **four U.S. states**:  

| Location | Count of Transactions |
|----------|------------------------|
| 🗽 New York (NY)   | 12,644 |
| 🌴 Florida (FL)    | 12,476 |
| 🌉 California (CA) | 12,468 |
| 🤠 Texas (TX)      | 12,412 |

#### 🔎 Insights
- The distribution is **fairly balanced** across all four states.  
- No state dominates the dataset, which helps in reducing **location bias** during fraud detection model training.  
- This balance ensures that the model does not unfairly associate fraud with one particular state due to skewed data.  

## Device Distribution  

| Device  | Count  |
|---------|--------|
| ATM     | 16,846 |
| Web     | 16,624 |
| Mobile  | 16,530 |

### 📱 Key Insight  
- Transactions are **well-balanced** across devices.  
- **ATM** is the most used channel, followed by **Web** and **Mobile**, but the differences are small.  
- This balance reduces the risk of **device-type bias** in modeling.  


## Class Distribution – IsFraud  

| Class | Count  |
|-------|--------|
| 0 (Non-Fraud) | 48,971 |
| 1 (Fraud)     | 1,029  |

### ⚖️ Key Insight  
- The dataset is **highly imbalanced**, with only **~2% fraudulent transactions**.  
- This imbalance requires careful handling, e.g., **resampling techniques (SMOTE/undersampling)** or **class-weighted models** for accurate fraud detection.  

## Feature Correlation  

| Feature        | TransactionID | Amount    | Time      | IsFraud   |
|----------------|---------------|----------|-----------|-----------|
| TransactionID  | 1.000         | 0.0029   | 0.0015    | 0.0007    |
| Amount         | 0.0029        | 1.000    | -0.0049   | -0.0022   |
| Time           | 0.0015        | -0.0049  | 1.000     | -0.0035   |
| IsFraud        | 0.0007        | -0.0022  | -0.0035   | 1.000     |

### 📊 Key Insight  
- **No strong correlations** exist between features; all correlation values are **very close to 0**.  
- `TransactionID`, `Amount`, and `Time` have **negligible correlation with `IsFraud`**, suggesting that fraud is not directly predictable from these raw numerical features alone.  
- This indicates the need for **feature engineering or additional behavioral features** to improve fraud detection.  

## 🛠️ Data Preprocessing 

### 1. Outlier Detection  
- Outliers, especially in the `Amount` feature, can **distort statistical measures** and **bias machine learning models**.  
- Detecting and handling outliers ensures that the model learns from **typical transaction patterns** rather than extreme cases.  

### 2. Log Transformation  
- The `Amount` feature is highly **skewed** due to a few large transactions.  
- Applying a **log transformation** reduces skewness, stabilizes variance, and makes the data more suitable for most ML algorithms.  

### 3. Label Encoding  
- Categorical features like `Location` and `Device` are **non-numeric**, so ML models cannot interpret them directly.  
- **Label encoding** converts these features into numerical form while preserving **distinct categories**, enabling the model to process them effectively.  

## Undersampling  
- The dataset is **highly imbalanced**: only ~2% of transactions are fraudulent.  
- **Undersampling** reduces the number of majority class (non-fraud) samples to **balance the dataset**, which helps the model **learn patterns for the minority class** without being biased toward the majority

## 🛠️ Feature Importance

- Not all features contribute equally to predicting fraud. Some may be **redundant or irrelevant**, which can **increase model complexity** and reduce performance.  
- Using a **Random Forest model** with `SelectFromModel` allows us to automatically **identify the most important features** based on their contribution to reducing impurity.  
- **Benefits of this step:**  
  - **Reduces dimensionality**, simplifying the model and improving training speed.  
  - **Improves model performance** by focusing on features that have the strongest predictive power.  
  - **Helps interpretability**, so we know which features drive fraud detection the most.  
- The selected features are then used for **model training**, ensuring the model learns from the **most informative predictors** rather than noise.  

# Random Forest Classifier Report (with class_weight='balanced')

**Precision (Fraud Class):** 0.022  
**Recall (Fraud Class):** 0.213  

**Confusion Matrix:**

|                | Predicted Non-Fraud | Predicted Fraud |
|----------------|------------------|----------------|
| **Actual Non-Fraud** | 11666            | 3014           |
| **Actual Fraud**     | 252              | 68             |

**Insights:**
- Model detects ~21% of fraud cases correctly (recall).  
- Precision is low due to many false positives.  
- SMOTE or threshold tuning can improve performance.

# Feature Importance
The plot below shows how much each feature contributes to the Random Forest model's decisions (after SMOTE oversampling):

- **Amount:** 0.484 → Most important feature, strongly influences fraud prediction.  
- **Time:** 0.430 → Second most important, moderately important for model.  
- **Location:** 0.043 → Minimal contribution.  
- **Device:** 0.043 → Minimal contribution.  

> Features with higher importance have a larger impact on the model's predictions.
