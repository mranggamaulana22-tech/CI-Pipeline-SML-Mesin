import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import dagshub
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
import os

# 1. Inisialisasi DagsHub
REPO_OWNER = "mranggamaulana22"
REPO_NAME = "Eksperimen_SML_Rangga-Maulana-Insani"
dagshub.init(repo_owner=REPO_OWNER, repo_name=REPO_NAME, mlflow=True)

print("[INFO] Mengambil dataset hasil preprocessing...")
# 2. Load dataset hasil preprocessing
data_path = "mesin_preprocessed.csv"
df_prep = pd.read_csv(data_path)

X_prep = df_prep.drop(columns=['Target'])
y_prep = df_prep['Target']

# 3. Setup Model & Hyperparameter Grid
rf = RandomForestClassifier(random_state=42, class_weight='balanced')
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5]
}

print("[INFO] Memulai training dan hyperparameter tuning dengan MLflow...")
with mlflow.start_run(run_name="RandomForest_Predictive_Maintenance"):
    # 4. Training dengan GridSearchCV
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=1)
    grid_search.fit(X_prep, y_prep)
    
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_prep)
    
    # 5. Manual Logging: Parameter, Metrik, dan Model
    mlflow.log_params(grid_search.best_params_) 
    
    acc = accuracy_score(y_prep, y_pred)
    mlflow.log_metric("training_accuracy", acc) 
    
    mlflow.sklearn.log_model(best_model, "random_forest_model") 
    
    print(f"[SUCCESS] Best Params: {grid_search.best_params_}")
    print(f"[SUCCESS] Training Accuracy: {acc:.4f}")

    # 6. Manual Logging: Artefak 1 (Confusion Matrix)
    cm = confusion_matrix(y_prep, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    cm_path = "confusion_matrix.png"
    plt.savefig(cm_path)
    mlflow.log_artifact(cm_path)
    plt.close()

    # 7. Manual Logging
    importances = best_model.feature_importances_
    plt.figure(figsize=(8, 6))
    pd.Series(importances, index=X_prep.columns).sort_values(ascending=True).plot(kind='barh', color='teal')
    plt.title("Feature Importance")
    fi_path = "feature_importance.png"
    plt.savefig(fi_path)
    mlflow.log_artifact(fi_path)
    plt.close()