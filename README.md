# Drug Response Analysis

This repository contains Drug Response Analysis . The goal is to analyze gene expression and clinical metadata to predict treatment response (Responder vs. Non-Responder) in an autoimmune disease study.

## Table of Contents
1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Data Description](#data-description)
4. [Approach Summary](#approach-summary)
   1. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
   2. [Feature Selection](#feature-selection)
   3. [Predictive Modeling](#predictive-modeling)
   4. [Model Explainability](#model-explainability)
   5. [Key Findings](#key-findings)
5. [How to Run](#how-to-run)
   1. [Environment Setup](#environment-setup)
   2. [Running the Code](#running-the-code)
   3. [Running Unit Tests](#running-unit-tests)
6. [Dependencies](#dependencies)


---

## Overview

We have:
- **`gene_expression.csv`**: Contains gene expression levels. Rows are samples, columns are genes.
- **`meta_data.csv`**: Contains metadata such as SampleID, Response (binary), DAS28 clinical score, and Gender.

We aim to:
1. Perform an **EDA** to understand the dataset distribution and handle missing data.
2. Identify top features (genes) that are most predictive of response.
3. Train a **classification model** (XGBoost) to predict treatment response.
4. Evaluate performance using accuracy, precision, recall, F1-score, and confusion matrix.
5. Demonstrate **explainability** with SHAP values and Feature importance .
6. Provide **unit tests** 

---

## Project Structure
```
Drug_Response_Analysis/
├── data/
│   ├── gene_expression.csv
│   └── meta_data.csv
├── src/
│   ├── main.py
│   ├── preprocessing/
│   │   ├── preprocess.py
│   │   ├── feature_selection.py
│   │   └── __init__.py
│   ├── models/
│   │   ├── model_training.py
│   │   ├── model_evaluation.py
│   │   └── __init__.py
│   ├── utils/
│   │   ├── config.py
│   │   └── __init__.py
├── tests/
│   ├── test_preprocess.py
│   ├── test_feature_selection.py
│   ├── test_model_training.py
│   ├── test_evaluation.py
├── run_tests.py
├── requirements.txt
└── README.md
```


**Key Python scripts**:
- `src/preprocessing/preprocess.py`  
  - Data loading, merging, preparation, scaling.
- `src/preprocessing/feature_selection.py`  
  - Functions to perform Lasso-based feature selection.
- `src/models/model_training.py`  
  - Functions to train and save XGBoost as baseline
- `src/models/model_evaluation.py`  
  - Evaluation metrics, confusion matrix plotting, feature imporantace and SHAP explainability.
- `src/main.py`  
  - Orchestrates the entire pipeline (optional entry point).

**Unit tests** are located in the `tests/` directory, each testing a separate module.

---

## Approach Summary

### 1. Exploratory Data Analysis (EDA)

**Data Loading**: Read both CSV files into Pandas DataFrames.  
**Merging**: Match samples by SampleID into a single combined DataFrame.  
**Missing Data**:

Checked for null values in gene expression and metadata (especially Response).  
Rows with missing Response are dropped if needed to keep the classification label consistent.  
we lots almost half of the data points  

**Distributions**:

Inspected gene expression distributions and the proportion of Responders vs. Non-Responders.  

**Outliers**:

Basic checks were conducted, and no extreme outliers were found, so the data was retained as is.

### 2. Feature Selection

Used Lasso (L1) Logistic Regression to identify top genes associated with the binary response:

- Split data into X (all genes + selected metadata) and y (Response).  
- Standard-scaled continuous features.  
- Fit a Lasso Logistic Regression with cross-validation.  
- Extract non-zero coefficients as "significant" genes.  

Returned the top 10 genes with the highest absolute coefficients.  

The approach was inspired by this reference:  
#https://datascience.stackexchange.com/questions/12455/feature-selection-for-gene-expression-dataset  

The 10 chosen features are:

```
['225591_at',
 '218430_s_at',
 '242842_at',
 '1558887_at',
 '225187_at',
 '216573_at',
 '238141_s_at',
 '1559434_at',
 '1566748_at',
 '223878_at']
```

It's interesting to note that their names follow a similar pattern. In the EDA part, it would be valuable to examine if they cluster together based on gene names.

### 3. Predictive Modeling

**The model explored**:

**XGBoost Classifier**:  
- Handles non-linearities and often performs well in tabular data tasks.  
- Trained with default parameters. Future work will include cross-validation and hyperparameter tuning for optimal performance.  

**Training/Validation**:  
- An 80/15 train/validation split was used. For future work, cross-validation or LOO (we have a small dataset) could be good.  
- Metrics computed:  
  - Accuracy  
  - Precision/Recall/F1  

Note that since the data was balanced, accuracy was an appropriate metric.  

**Confusion Matrix**:  
- Visualized how many Responders vs. Non-Responders were correctly/incorrectly classified.  

### 4. Model Explainability

- As we used a tree-based model, I easily extracted the most significant features from both the 10 selected features from the previous task and the metadata features, which were found to not contribute significantly.  

- Used SHAP to interpret individual predictions:  
  - Calculated SHAP values for each feature to see which contribute most to predicted response.  
  - Provided force plots and waterfall plots for explanation.  

### 5. Key Findings

- **Top Genes**: Listed 10 genes with highest absolute Lasso coefficients.  
- **Model Performance**: Accuracy and F1 score approximately 0.60%. Since the data is balanced, precision and recall are at similar values. I should note that we clearly have overfitting (common with XGBoost without proper hyperparameter tuning) and could likely achieve better results. As this is a baseline solution for the task, I kept it as is.  
- **Clinical Relevance**: Most genes were less relevant in this project, and the metadata features weren't helpful for classification.  

---

# How to Run

## 1. Environment Setup

### Clone this repository:
```
git clone https://github.com/arieloren/Drug_Response_Analysis.git
cd Drug_Response_Analysis
```

### Install dependencies:
```
pip install -r requirements.txt
```

---

## 2. Running the Code

### (Optional) Update File Paths
If needed, update file paths in `src/utils/config.py` or wherever the CSV paths are defined.

### Run the Main Pipeline
`main.py` orchestrates the entire process:
```
python src/main.py
```
This will:
- Load the data
- Merge, preprocess, and select features
- Train and evaluate the model
- Print the results

---

## 3. Running Unit Tests

We use `unittest` for testing. To run all tests, use the `run_tests.py` script from the project root:
```
$env:PYTHONPATH="src"; python -m unittest discover -s tests
```

This will execute all tests in the `tests/` folder, including:
- Preprocessing tests
- Feature selection tests
- Model training tests
- Evaluation tests

---
### ⚠ Known Issue  

There is an issue with `test_xgboost_training`, specifically with:  

```python
self.assertEqual(model.n_estimators, 100)
```

Even though 100 is the initial parameter for XGBoost, the assertion fails.
I haven't had time to investigate the cause yet, so please be aware of this when running tests.

## Dependencies

All required packages are listed in `requirements.txt`. The core dependencies include:

- **Python** >= 3.8
- **pandas**
- **numpy**
- **scikit-learn**
- **xgboost**
- **shap**
- **matplotlib**
- **seaborn**
- **joblib**
- **plotly**
- `unittest` (included in Python’s standard library, no separate installation needed)

