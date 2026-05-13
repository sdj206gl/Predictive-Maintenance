
import pandas as pd
import numpy as np
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    BaggingClassifier, RandomForestClassifier,
    AdaBoostClassifier, GradientBoostingClassifier
)
from xgboost import XGBClassifier

import mlflow
import mlflow.sklearn
import mlflow.xgboost
from mlflow.models.signature import infer_signature

from huggingface_hub import HfApi

def load_data_from_hf():
    print("Loading datasets from HuggingFace Hub...")
    X_train = pd.read_csv("hf://datasets/shashidj/Predictive-Maintenance/X_train.csv")
    X_test  = pd.read_csv("hf://datasets/shashidj/Predictive-Maintenance/X_test.csv")
    y_train = pd.read_csv("hf://datasets/shashidj/Predictive-Maintenance/y_train.csv").squeeze()
    y_test  = pd.read_csv("hf://datasets/shashidj/Predictive-Maintenance/y_test.csv").squeeze()
    print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
    return X_train, X_test, y_train, y_test

MODEL_CONFIGS = {
    'DecisionTree': {
        'model': DecisionTreeClassifier(random_state=42),
        'params': {
            'max_depth': [3, 5, 10, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    },
    'RandomForest': {
        'model': RandomForestClassifier(random_state=42, n_jobs=-1),
        'params': {
            'n_estimators': [100, 200, 300],
            'max_depth': [5, 10, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
    },
    'Bagging': {
        'model': BaggingClassifier(random_state=42, n_jobs=-1),
        'params': {
            'n_estimators': [10, 50, 100],
            'max_samples': [0.8, 1.0],
            'max_features': [0.8, 1.0]
        }
    },
    'AdaBoost': {
        'model': AdaBoostClassifier(random_state=42),
        'params': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 1.0]
        }
    },
    'GradientBoosting': {
        'model': GradientBoostingClassifier(random_state=42),
        'params': {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        }
    },
    'XGBoost': {
        'model': XGBClassifier(random_state=42, eval_metric='logloss'),
        'params': {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 1.0]
        }
    }
}

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    metrics = {
        'accuracy':  accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall':    recall_score(y_test, y_pred, average='weighted'),
        'f1_score':  f1_score(y_test, y_pred, average='weighted'),
        'roc_auc':   roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else 0
    }
    return metrics, y_pred

def train_and_evaluate_models():
    X_train, X_test, y_train, y_test = load_data_from_hf()
    mlflow.set_experiment("Predictive Maintenance")

    best_model, best_score, best_model_name = None, 0, ""
    results_summary = []

    print("\n🚀 Starting Model Training and Hyperparameter Tuning...\n")
    for model_name, config in MODEL_CONFIGS.items():
        print(f"Training {model_name}...")
        with mlflow.start_run(run_name=f"{model_name}_tuning"):
            grid_search = GridSearchCV(
                estimator=config['model'], param_grid=config['params'],
                scoring='accuracy', cv=5, n_jobs=-1, verbose=1
            )
            grid_search.fit(X_train, y_train)
            best_estimator = grid_search.best_estimator_

            mlflow.log_params(grid_search.best_params_)
            mlflow.log_param("model_type", model_name)
            mlflow.log_param("cv_folds", 5)

            metrics, y_pred = evaluate_model(best_estimator, X_test, y_test)
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)

            cv_scores = cross_val_score(best_estimator, X_train, y_train, cv=5)
            mlflow.log_metric("cv_mean_accuracy", cv_scores.mean())
            mlflow.log_metric("cv_std_accuracy", cv_scores.std())

            signature = infer_signature(X_train, best_estimator.predict(X_train))
            if model_name == 'XGBoost':
                mlflow.xgboost.log_model(best_estimator, "model", signature=signature)
            else:
                mlflow.sklearn.log_model(best_estimator, "model", signature=signature)

            joblib.dump(best_estimator, f"{model_name}_model.pkl")

            if metrics['accuracy'] > best_score:
                best_score = metrics['accuracy']
                best_model = best_estimator
                best_model_name = model_name

            results_summary.append({
                'Model': model_name,
                'Best_Params': grid_search.best_params_,
                'CV_Score': cv_scores.mean(),
                'Test_Accuracy': metrics['accuracy'],
                'Test_F1': metrics['f1_score'],
                'Test_ROC_AUC': metrics['roc_auc']
            })
            print(f"✅ {model_name} - Test Accuracy: {metrics['accuracy']:.4f}")

    results_df = pd.DataFrame(results_summary).sort_values('Test_Accuracy', ascending=False)
    results_df.to_csv("model_comparison_results.csv", index=False)
    print("\n🏆 MODEL COMPARISON RESULTS:")
    print(results_df.to_string(index=False))
    print(f"\n🥇 BEST MODEL: {best_model_name} with Accuracy: {best_score:.4f}")
    joblib.dump(best_model, "best_model.pkl")
    return best_model, best_model_name, results_df

def register_best_model_to_hf(model, model_name, X_test, y_test):
    print(f"\n📦 Registering {model_name} to HuggingFace Hub...")
    model_dir = "predictive_maintenance_best_model"
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, f"{model_dir}/model.pkl")

    model_card = f"""---
library_name: scikit-learn
tags:
- predictive-maintenance
- classification
- mlops
- {model_name.lower()}
---

# Predictive Maintenance Model

## Model Description
This is a {model_name} model trained to predict engine failure based on sensor readings.

## Model Performance
- Algorithm: {model_name}
- Task: Binary Classification
- Dataset: Engine Condition Sensor Data
- Accuracy: {accuracy_score(y_test, model.predict(X_test)):.4f}

## Usage
```python
import joblib
model = joblib.load('model.pkl')
predictions = model.predict(X)
```

## Features
Engine RPM, Lub Oil Pressure, Fuel Pressure, Coolant Pressure, Lub Oil Temp, Coolant Temp
"""
    with open(f"{model_dir}/README.md", "w") as f:
        f.write(model_card)

    api = HfApi(token=os.getenv("HF_TOKEN"))
    try:
        api.create_repo(repo_id="shashidj/Predictive-Maintenance-Model", repo_type="model", exist_ok=True)
        api.upload_folder(folder_path=model_dir, repo_id="shashidj/Predictive-Maintenance-Model", repo_type="model")
        print("✅ Model successfully registered to HuggingFace Hub!")
        print("🔗 Model URL: https://huggingface.co/shashidj/Predictive-Maintenance-Model")
    except Exception as e:
        print(f"❌ Error uploading to HuggingFace: {e}")

if __name__ == "__main__":
    best_model, best_model_name, results_df = train_and_evaluate_models()
    _, X_test, _, y_test = load_data_from_hf()
    register_best_model_to_hf(best_model, best_model_name, X_test, y_test)
    print("\n🎉 Model Building and Registration Complete!")
    print("📊 Check MLflow UI for detailed experiment tracking")
