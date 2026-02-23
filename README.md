# 💰 Credit Card Score Prediction - Financial ML

A **machine learning model for credit card scoring** using advanced feature engineering and ensemble methods to predict creditworthiness, enabling financial institutions to make informed lending decisions with high accuracy.

## 🎯 Overview

This project demonstrates:
- ✅ Feature engineering for financial data
- ✅ Handling imbalanced datasets
- ✅ Multiple ML algorithms
- ✅ Model evaluation metrics
- ✅ Credit risk assessment
- ✅ Threshold optimization

## 📊 Dataset Overview

```
Features: 31
Target: Credit Score (0-200)
Samples: 100,000+ customers
Balance: ~60% good credit, ~40% poor credit
```

## 🔧 Features

### Demographic Features
```
- Age (18-82)
- Gender
- Marital Status
- Education Level
- Employment Status
```

### Financial Features
```
- Annual Income
- Monthly Salary
- Existing Loans
- Credit History Length
- Payment History
```

### Behavioral Features
```
- Number of Accounts
- Credit Utilization Rate
- Payment Delays
- Recent Inquiries
- Delinquency Status
```

## 🛠️ Feature Engineering

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class CreditFeatureEngineer:
    def __init__(self, data):
        self.data = data
    
    def create_derived_features(self):
        """Create new features from existing ones"""
        
        # Income-to-debt ratio
        self.data['income_debt_ratio'] = (
            self.data['annual_income'] / 
            (self.data['existing_loans'] + 1)
        )
        
        # Credit utilization (new vs existing credit)
        self.data['credit_ratio'] = (
            self.data['used_credit'] / 
            (self.data['total_credit_limit'] + 1)
        )
        
        # Payment history score
        self.data['on_time_payment_rate'] = (
            self.data['on_time_payments'] / 
            (self.data['total_payments'] + 1)
        )
        
        # Account diversity
        self.data['account_diversity'] = (
            self.data['credit_cards'] + 
            self.data['loans'] + 
            self.data['mortgages']
        )
        
        # Credit age (months)
        self.data['credit_age'] = (
            (pd.Timestamp.now() - self.data['credit_start_date']).dt.days / 30
        )
        
        return self.data
    
    def handle_categorical(self):
        """Encode categorical variables"""
        categorical_cols =  ['gender', 'marital_status', 'education']
        
        # Ordinal encoding for ordered categories
        education_mapping = {
            'High School': 1,
            'Bachelor': 2,
            'Master': 3,
            'PhD': 4
        }
        self.data['education_encoded'] = self.data['education'].map(education_mapping)
        
        # One-hot encoding for nominal categories
        self.data = pd.get_dummies(self.data, columns=['gender', 'marital_status'])
        
        return self.data
```

## 🤖 Model Development

### Ensemble Approach

```python
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier
)
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import lightgbm as lgb

class CreditScoreEnsemble:
    def __init__(self):
        self.models = {
            'logistic': LogisticRegression(max_iter=1000),
            'random_forest': RandomForestClassifier(n_estimators=100),
            'gradient_boost': GradientBoostingClassifier(n_estimators=100),
            'xgboost': XGBClassifier(n_estimators=100),
            'lightgbm': lgb.LGBMClassifier(n_estimators=100),
            'adaboost': AdaBoostClassifier(n_estimators=100)
        }
        self.ensemble_weights = None
    
    def train(self, X_train, y_train):
        """Train all models"""
        for name, model in self.models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
    
    def predict_ensemble(self, X_test):
        """Weighted ensemble prediction"""
        predictions = np.zeros(len(X_test))
        
        for name, model in self.models.items():
            weight = self.ensemble_weights[name]
            pred = model.predict_proba(X_test)[:, 1]
            predictions += weight * pred
        
        return (predictions > 0.5).astype(int)
    
    def optimize_weights(self, X_val, y_val):
        """Find optimal ensemble weights"""
        from scipy.optimize import minimize
        
        def objective(weights):
            predictions = np.zeros(len(X_val))
            norm_weights = weights / weights.sum()
            
            for (name, model), weight in zip(self.models.items(), norm_weights):
                pred = model.predict_proba(X_val)[:, 1]
                predictions += weight * pred
            
            pred_binary = (predictions > 0.5).astype(int)
            return -accuracy_score(y_val, pred_binary)
        
        initial_weights = np.ones(len(self.models)) / len(self.models)
        result = minimize(objective, initial_weights)
        
        self.ensemble_weights = result.x / result.x.sum()
```

## 📈 Model Evaluation

### Performance Metrics

```python
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve
)

class ModelEvaluator:
    @staticmethod
    def evaluate(y_true, y_pred):
        """Comprehensive model evaluation"""
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred),
        }
        
        cm = confusion_matrix(y_true, y_pred)
        
        # Confusion matrix breakdown
        tn, fp, fn, tp = cm.ravel()
        
        metrics['true_negative_rate'] = tn / (tn + fp)
        metrics['false_positive_rate'] = fp / (fp + tn)
        metrics['false_negative_rate'] = fn / (fn + tp)
        
        return metrics
    
    @staticmethod
    def evaluate_probabilities(y_true, y_proba):
        """Evaluate probabilistic predictions"""
        
        auc = roc_auc_score(y_true, y_proba)
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        
        return {
            'auc': auc,
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds
        }
```

### Threshold Optimization

```python
class ThresholdOptimizer:
    """Optimize decision threshold for specific business needs"""
    
    @staticmethod
    def find_optimal_threshold(y_true, y_proba, metric='f1'):
        """Find threshold that maximizes metric"""
        
        best_score = 0
        best_threshold = 0.5
        
        for threshold in np.arange(0, 1, 0.01):
            y_pred = (y_proba >= threshold).astype(int)
            
            if metric == 'f1':
                score = f1_score(y_true, y_pred)
            elif metric == 'precision':
                score = precision_score(y_true, y_pred)
            elif metric == 'recall':
                score = recall_score(y_true, y_pred)
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        return best_threshold, best_score
    
    @staticmethod
    def find_business_optimal_threshold(y_true, y_proba, 
                                       cost_fp=1,  # Cost of false positive
                                       cost_fn=5):  # Cost of false negative
        """Optimize threshold for business costs"""
        
        best_cost = float('inf')
        best_threshold = 0.5
        
        for threshold in np.arange(0, 1, 0.01):
            y_pred = (y_proba >= threshold).astype(int)
            cm = confusion_matrix(y_true, y_pred)
            tn, fp, fn, tp = cm.ravel()
            
            total_cost = (fp * cost_fp) + (fn * cost_fn)
            
            if total_cost < best_cost:
                best_cost = total_cost
                best_threshold = threshold
        
        return best_threshold, best_cost
```

## ⚖️ Handling Imbalanced Data

```python
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

class BalancingStrategy:
    @staticmethod
    def smote_strategy(X_train, y_train):
        """SMOTE - Synthetic Minority Over-sampling"""
        smote = SMOTE(sampling_strategy=0.5)
        X_balanced, y_balanced = smote.fit_resample(X_train, y_train)
        return X_balanced, y_balanced
    
    @staticmethod
    def combined_strategy(X_train, y_train):
        """Combine over and under sampling"""
        pipeline = Pipeline([
            ('over', SMOTE(sampling_strategy=0.5)),
            ('under', RandomUnderSampler(sampling_strategy=0.8))
        ])
        X_balanced, y_balanced = pipeline.fit_resample(X_train, y_train)
        return X_balanced, y_balanced
    
    @staticmethod
    def class_weights(y_train):
        """Use class weights in model"""
        from sklearn.utils.class_weight import compute_class_weight
        
        weights = compute_class_weight(
            'balanced',
            np.unique(y_train),
            y_train
        )
        return dict(enumerate(weights))
```

## 💡 Interview Talking Points

**Q: How do you handle imbalanced credit data?**
```
Answer:
1. SMOTE - synthetic minority samples
2. Class weights - penalize minority mistakes
3. Stratified sampling - preserve class ratio
4. Different thresholds - business-specific optimization
```

**Q: What metrics matter most for credit scoring?**
```
Answer:
- Recall (catch all defaulters)
- Precision (avoid unnecessary rejections)
- F1 (balanced trade-off)
- Business cost analysis
```

## 🌟 Portfolio Value

✅ Financial ML expertise
✅ Imbalanced data handling
✅ Ensemble methods
✅ Model evaluation metrics
✅ Threshold optimization
✅ Business-oriented ML

## 📄 License

MIT License - Educational Use

---

**Technologies**: Scikit-learn, XGBoost, LightGBM, Imbalanced-learn

