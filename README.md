# 📊 Credit Card Score Prediction - Financial ML

A **machine learning system for predicting credit card scores** using historical financial data, classification models, and feature engineering.

## 🎯 Overview

This project covers:
- ✅ Financial data analysis
- ✅ Feature engineering
- ✅ Classification models
- ✅ Model evaluation
- ✅ Risk assessment
- ✅ Interpretability
- ✅ Business insights

## 💳 Data Preprocessing

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

class CreditDataProcessor:
    """Process credit card data"""
    
    def __init__(self):
        self.data = None
        self.scaler = StandardScaler()
    
    def load_data(self, filepath):
        """Load credit dataset"""
        self.data = pd.read_csv(filepath)
        print(f"Dataset shape: {self.data.shape}")
        return self.data
    
    def handle_missing_values(self):
        """Handle missing data"""
        df = self.data.copy()
        
        # Missing value percentage
        missing_pct = (df.isnull().sum() / len(df)) * 100
        print(f"Missing values:\n{missing_pct[missing_pct > 0]}")
        
        # Drop columns with >50% missing
        df = df.dropna(thresh=0.5 * len(df), axis=1)
        
        # Impute numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            df[col].fillna(df[col].median(), inplace=True)
        
        return df
    
    def feature_engineering(self):
        """Create financial features"""
        df = self.data.copy()
        
        # Income-based features
        if 'annual_income' in df.columns:
            df['income_category'] = pd.qcut(df['annual_income'], 
                                            q=4, 
                                            labels=['low', 'medium', 'high', 'very_high'])
        
        # Payment behavior
        if 'num_late_payments' in df.columns:
            df['late_payment_ratio'] = df['num_late_payments'] / df['account_age_months']
        
        # Utilization ratio
        if 'credit_limit' in df.columns and 'balance' in df.columns:
            df['utilization_ratio'] = df['balance'] / df['credit_limit']
        
        # Debt-to-income
        if 'total_debt' in df.columns and 'annual_income' in df.columns:
            df['debt_to_income'] = df['total_debt'] / df['annual_income']
        
        # Account age categories
        if 'account_age_months' in df.columns:
            df['account_age_category'] = pd.cut(df['account_age_months'],
                                               bins=[0, 12, 36, 60, 120, np.inf],
                                               labels=['new', 'young', 'established', 'mature', 'old'])
        
        # Payment history score
        if 'on_time_payments' in df.columns and 'total_payments' in df.columns:
            df['payment_reliability'] = df['on_time_payments'] / (df['total_payments'] + 1)
        
        return df
    
    def encode_categorical(self):
        """Encode categorical features"""
        df = self.data.copy()
        
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
        
        return df
    
    def normalize_features(self, X_train):
        """Scale numerical features"""
        X_scaled = self.scaler.fit_transform(X_train)
        
        return X_scaled
```

## 🤖 Classification Models

```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV

class CreditScoreClassifier:
    """Predict credit score category"""
    
    def __init__(self):
        self.models = {}
        self.best_model = None
    
    def logistic_regression(self, X_train, y_train):
        """Baseline model"""
        lr = LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight='balanced'
        )
        
        lr.fit(X_train, y_train)
        self.models['lr'] = lr
        
        return lr
    
    def random_forest(self, X_train, y_train):
        """Random Forest"""
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        )
        
        rf.fit(X_train, y_train)
        self.models['rf'] = rf
        
        return rf
    
    def gradient_boosting(self, X_train, y_train):
        """Gradient Boosting"""
        gb = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=7,
            min_samples_split=5,
            subsample=0.8,
            random_state=42
        )
        
        gb.fit(X_train, y_train)
        self.models['gb'] = gb
        
        return gb
    
    def hyperparameter_tuning(self, X_train, y_train):
        """Grid search for best params"""
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15],
            'learning_rate': [0.01, 0.1, 0.2]
        }
        
        gb = GradientBoostingClassifier(random_state=42)
        
        grid_search = GridSearchCV(
            gb,
            param_grid,
            cv=5,
            scoring='roc_auc_ovr',
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"Best params: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.3f}")
        
        self.best_model = grid_search.best_estimator_
        
        return self.best_model
```

## 📈 Model Evaluation

```python
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, auc, precision_recall_curve
)

class CreditModelEvaluator:
    """Evaluate model performance"""
    
    @staticmethod
    def evaluate(model, X_test, y_test):
        """Comprehensive evaluation"""
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        metrics = {
            'roc_auc': roc_auc_score(y_test, y_pred_proba, multi_class='ovr'),
            'classification_report': classification_report(y_test, y_pred)
        }
        
        print(metrics['classification_report'])
        
        return metrics
    
    @staticmethod
    def feature_importance(model, feature_names):
        """Feature importance analysis"""
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            print("Feature Rankings:")
            for i in range(min(10, len(feature_names))):
                print(f"{i+1}. {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")
            
            return importances
    
    @staticmethod
    def plot_roc_curve(model, X_test, y_test):
        """ROC curve visualization"""
        y_pred_proba = model.predict_proba(X_test)
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
        roc_auc = auc(fpr, tpr)
        
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.show()
```

## 🎯 Risk Scoring

```python
class CreditRiskScorer:
    """Calculate risk scores"""
    
    @staticmethod
    def calculate_risk_score(model, features):
        """Predict risk probability"""
        risk_prob = model.predict_proba([features])[0]
        
        # Assuming 0=Good, 1=Poor
        risk_score = risk_prob[1] * 100
        
        # Risk category
        if risk_score < 20:
            risk_category = 'Low Risk'
        elif risk_score < 50:
            risk_category = 'Medium Risk'
        else:
            risk_category = 'High Risk'
        
        return {
            'risk_score': risk_score,
            'risk_category': risk_category,
            'good_probability': risk_prob[0]
        }
    
    @staticmethod
    def generate_recommendation(risk_score):
        """Business recommendation"""
        if risk_score < 20:
            return "Approve with standard terms"
        elif risk_score < 50:
            return "Approve with monitoring"
        else:
            return "Request additional documentation or deny"
```

## 💡 Interview Talking Points

**Q: Class imbalance in credit scoring?**
```
Answer:
- Good credit cards common, bad rare
- Use balanced class weights
- SMOTE oversampling
- ROC-AUC vs accuracy metric
- Cost matrix implementation
```

**Q: Feature importance in financial?**
```
Answer:
- Payment history critical
- Utilization ratio matters
- Income stability useful
- Account age proxy for reliability
- SHAP/LIME for transparency
```

## 🌟 Portfolio Value

✅ Financial data analysis
✅ Feature engineering
✅ Classification models
✅ Hyperparameter tuning
✅ ROC-AUC evaluation
✅ Risk scoring
✅ Business recommendations

---

**Technologies**: Scikit-learn, Pandas, XGBoost

