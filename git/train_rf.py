import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import time
from pathlib import Path

def train_rf_model(dataset_path: str, output_dir: str = "ml_results"):
    """
    Train a Random Forest model to detect Terraform misconfigurations.
    """
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    print(f"📂 Loading labeled dataset: {dataset_path}")
    df = pd.read_csv(dataset_path)
    
    # --- 1. DATA PREPROCESSING ---
    print("🛠️  Preprocessing data...")
    
    # Columns to exclude from features
    exclude_cols = [
        'file_path', 'resource_type', 'resource_name', 
        'kics_has_issues', 'kics_total_issues', 'kics_severity_high', 
        'kics_severity_medium', 'kics_severity_low', 'kics_categories', 
        'kics_query_ids'
    ]
    
    # Identify target and features
    target = 'kics_has_issues'
    features = [col for col in df.columns if col not in exclude_cols]
    
    print(f"   Features being used ({len(features)}): {', '.join(features)}")
    
    X = df[features].copy()
    y = df[target].copy()
    
    # Handle missing values (fill with 0 for binary flags, median for counts)
    for col in X.columns:
        if X[col].dtype == 'bool' or X[col].dtype == 'object':
            X[col] = X[col].fillna(False).astype(int)
        else:
            X[col] = X[col].fillna(X[col].median())

    # Map target booleans to integers
    y = y.astype(int)
    
    # --- 2. TRAIN-TEST SPLIT ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"📊 Dataset split: Train={len(X_train):,}, Test={len(X_test):,}")

    # --- 3. TRAINING ---
    print(f"🚀 Training Random Forest model with 100 trees...")
    start_time = time.time()
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    elapsed = time.time() - start_time
    print(f"✅ Training complete in {elapsed:.1f}s")
    
    # --- 4. EVALUATION ---
    print("\n📈 Evaluating model performance...")
    y_pred = rf.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    
    print(f"   Accuracy: {acc:.4f}")
    print("\n   Confusion Matrix:")
    print(conf_matrix)
    print("\n   Classification Report:")
    print(class_report)
    
    # --- 5. FEATURE IMPORTANCE ---
    importances = pd.DataFrame({
        'feature': features,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n🌟 Top 10 Feature Importances:")
    print(importances.head(10))
    
    # --- 6. SAVE RESULTS ---
    importances.to_csv(f"{output_dir}/rf_feature_importances.csv", index=False)
    joblib.dump(rf, f"{output_dir}/rf_model.joblib")
    
    print(f"\n📁 Results saved to {output_dir}/ directory")

if __name__ == "__main__":
    dataset = "/home/swarna/Downloads/mainproo/dataset_new/sub-data/terraform_dataset_labeled.csv"
    train_rf_model(dataset)
