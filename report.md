
# Credit Card Behaviour Score Model Report

## Problem Statement
The goal is to develop a predictive model for the probability of credit card default using the provided development data.

## Methodology
### 1. Exploratory Data Analysis
- Checked for missing values and imputed them using the median strategy.
- Analyzed the distribution of the target variable (`bad_flag`).

### 2. Feature Engineering
- Imputed missing values using median.
- Standardized numerical features for uniform scaling.

### 3. Model Development
- Models evaluated: GradientBoosting, XGBoost, and LightGBM.
- **Best Model**: XGBoost with an ROC-AUC score of 0.7730.

### 4. Feature Importance
Top contributing features:
Feature  Importance
     f1        70.0
     f4        59.0
     f2        33.0
    f25        21.0
    f79        17.0
    f23        16.0
    f82        16.0
     f8        15.0
     f6        14.0
    f70        14.0

### 5. Validation Predictions
Predicted probabilities saved to `validation_predictions.csv`.

## Evaluation Metric
- Test ROC-AUC Score: 0.7730

## Insights
- The most important features include: f1, f4, f2.
- Standardizing features improved model performance.
