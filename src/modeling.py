import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import fbeta_score, roc_auc_score, accuracy_score, recall_score, confusion_matrix
from lightgbm import LGBMClassifier
from imblearn.combine import SMOTEENN
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
import joblib
import json
from utils import prepare_dataset

# Load and prepare the dataset
df = pd.read_csv('data/raw/cs-training.csv', index_col=0)
df = prepare_dataset(df)
df.to_csv('data/processed/processed_training_data.csv', index=False)

target = 'SeriousDlqin2yrs'
X = df.drop(columns=[target])
y = df[target]

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Balance the training data using SMOTEENN
smoteenn = SMOTEENN(random_state=42)
X_train_res, y_train_res = smoteenn.fit_resample(X_train, y_train)

# Define the Optuna objective function
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
        'max_depth': trial.suggest_int('max_depth', 4, 15),
        'num_leaves': trial.suggest_int('num_leaves', 16, 256),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.2),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 5.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 5.0),
        'class_weight': 'balanced',
        'random_state': 42,
        'n_jobs': -1
    }

    model = LGBMClassifier(**params)
    model.fit(X_train_res, y_train_res)
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    # Pick a threshold to convert probabilities to binary predictions
    threshold = trial.suggest_float('threshold', 0.1, 0.9)
    y_pred = (y_pred_prob >= threshold).astype(int)
    f2 = fbeta_score(y_test, y_pred, beta=2)
    return f2

# Run hyperparameter tuning
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

# Train the final model with the best parameters
best_params = study.best_params.copy()
best_threshold = best_params.pop('threshold')

model = LGBMClassifier(**best_params)
model.fit(X_train_res, y_train_res)

y_pred_prob = model.predict_proba(X_test)[:, 1]
y_pred = (y_pred_prob >= best_threshold).astype(int)

# Print evaluation metrics
print("Best threshold:", best_threshold)
print("F2-score:", fbeta_score(y_test, y_pred, beta=2))
print("ROC AUC:", roc_auc_score(y_test, y_pred_prob))
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Recall for the rare class:", recall_score(y_test, y_pred, pos_label=1))

# Plot the confusion matrix
cm = confusion_matrix(y_test, y_pred)
cm_percent = cm.astype('float') / cm.sum(axis=1)[:, None] * 100

plt.figure(figsize=(6,5))
sns.heatmap(cm_percent, annot=True, fmt=".1f", cmap="Blues", xticklabels=[0,1], yticklabels=[0,1])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Confusion Matrix (%) - LGBM (threshold={best_threshold:.2f})")
plt.show()

# Save the model and the threshold
joblib.dump(model, "models/lgbm_model.pkl")

with open("models/best_threshold.json", "w") as f:
    json.dump({"best_threshold": best_threshold}, f)







