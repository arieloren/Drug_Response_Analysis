# Drug Response Analysis: Gene Expression & Patient Metadata

This repository contains the solution to the take-home exam for a Data Scientist position. The goal is to analyze gene expression and clinical metadata to predict treatment response (Responder vs. Non-Responder) in an autoimmune disease study.

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
7. [Contact](#contact)

---

## Overview

We have:
- **`gene_expression.csv`**: Contains gene expression levels. Rows are samples, columns are genes.
- **`meta_data.csv`**: Contains metadata such as SampleID, Response (binary), DAS28 clinical score, and Gender.

We aim to:
1. Perform an **EDA** to understand the dataset distribution and handle missing data.
2. Identify top features (genes) that are most predictive of response.
3. Train a **classification model** (Logistic Regression with L1 regularization and/or XGBoost) to predict treatment response.
4. Evaluate performance using accuracy, precision, recall, F1-score, and confusion matrix.
5. Demonstrate **explainability** with SHAP values and Feature importance .
6. Provide **unit tests** demonstrating good coding practices.

---

## Project Structure

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

## Data Description

1. **gene_expression.csv**  
   - Each row corresponds to a patient sample.  
   - Columns include:
     - `ID_REF`: If present, a unique gene reference or probe ID.
     - `GENE1`, `GENE2`, ..., `GENE_N`: Expression values for each gene.

2. **meta_data.csv**  
   - `SampleID`: Unique sample/patient identifier (matches gene_expression).  
   - `Response`: Binary label (`Responder`, `Non_Responder`).  
   - `DAS28`: Disease Activity Score (continuous numeric).  
   - `Gender`: `Male` or `Female`.  

---

## Approach Summary

### 1. Exploratory Data Analysis (EDA)

- **Data Loading**: Read both CSV files into Pandas DataFrames.
- **Merging**: Match samples by `SampleID` into a single combined DataFrame.
- **Missing Data**:  
  - Checked for null values in gene expression and metadata (especially `Response`).
  - Rows with missing `Response` are dropped if needed to keep the classification label consistent.
- **Distributions**:  
  - Inspected gene expression distributions and the proportion of Responders vs. Non-Responders.
- **Outliers**:  
  - Basic checks; didnt find any extreme outleirs, so i kept the data

### 2. Feature Selection

- Used **Lasso (L1) Logistic Regression** to identify top genes associated with the binary response:
  1. Split data into `X` (all genes + selected metadata) and `y` (`Response`).
  2. Standard-scaled continuous features.
  3. Fit a Lasso Logistic Regression with cross-validation.
  4. Extract non-zero coefficients as “significant” genes.
- Returned the **top 10 genes** with the highest absolute coefficients.

### 3. Predictive Modeling

the modelsexplored:
2. **XGBoost Classifier**:
   - Handles non-linearities and often performs well in tabular data tasks.
   - Trained with default parameters future work will have cross-validation for best hyperparameters (optional).

**Training/Validation**:
- A train/validation split (e.g., 80/15) was used.
- Metrics computed:
  - **Accuracy**
  - **Precision/Recall/F1**
  but we have to see that the data was balanced so accuracy is a fit for the job

**Confusion Matrix**:
- Visualized how many Responders vs. Non-Responders were correctly/incorrectly classified.

### 4. Model Explainability
as we used a tree based model, i extracted easly the most sugnificat features from both the 10 selected features from prev task, and the metadata features, that i figure out didnt helpt much
- Used **SHAP** to interpret individual predictions:
  - Calculated SHAP values for each feature to see which contribute most to predicted response.
  - Provided force plots and waterfall plots for explanation.

### 5. Key Findings

- **Top Genes**: Listed 10 genes with highest absolute Lasso coefficients.
- **Model Performance**: Accuracy and F1 ~ XX%, with a certain trade-off between precision and recall.
- **Clinical Relevance**: Genes consistently associated with immune/inflammatory processes appear among the top features.

---

## How to Run

### 1. Environment Setup

1. **Clone** this repository:
   ```bash
   git clone https://github.com/YourUsername/Drug_Response_Analysis.git
   cd Drug_Response_Analysis

Create and activate a virtual environment (optional but recommended).
Install dependencies:
bash
Copy
Edit
pip install -r requirements.txt
2. Running the Code
(Optional) Update file paths in src/utils/config.py or wherever the CSV paths are defined, if needed.
Run the main pipeline (if main.py orchestrates everything):
bash
Copy
Edit
python src/main.py
This will:
Load the data
Merge, preprocess, feature-select
Train and evaluate the model
Print or save results
3. Running Unit Tests
We use unittest for testing. From the project root, run:

bash
Copy
Edit
# Option A: using the run_tests.py script
python run_tests.py

# Option B: via unittest discover
python -m unittest discover -s tests
This will run all tests in the tests/ folder:

Preprocessing tests
Feature Selection tests
Model Training tests
Evaluation tests
Dependencies
All required packages are listed in requirements.txt. The core ones include:

python >= 3.8
pandas
numpy
scikit-learn
xgboost
shap
matplotlib
seaborn
joblib
unittest (standard library, no separate install needed)

