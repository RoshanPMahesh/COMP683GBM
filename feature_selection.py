import pandas as pd
import numpy as np
import os
from pathlib import Path
import glob
import re
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from lightgbm import LGBMClassifier
from collections import Counter

def load_metadata(csv_path):
    """
    Load metadata from the CSV file and create a mapping of patient IDs to their treatment responses
    """
    print(f"Loading metadata from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"CSV columns: {df.columns.tolist()}")
    
    # Create a mapping of patient IDs to treatment responses
    metadata = {}
    id_column = 'Sample ID'
    
    # Calculate median overall survival
    #print(df['OS (days)'])
    median_os = df['OS (days)'].median()
    print(f"Median overall survival: {median_os} days")
    
    # Create treatment response classification
    for _, row in df.iterrows():
        patient_id = str(row[id_column]).strip()
        if not patient_id or patient_id == 'nan':
            continue
        
        os_days = row['OS (days)']
        if pd.isna(os_days):
            continue
            
        is_responder = os_days > median_os
        metadata[patient_id] = "Responder" if is_responder else "Non-responder"
    
    print(f"Loaded metadata for {len(metadata)} patients")
    print(f"Responders: {sum(1 for v in metadata.values() if v == 'Responder')}")
    print(f"Non-responders: {sum(1 for v in metadata.values() if v == 'Non-responder')}")
    
    print(metadata)
    return metadata

def extract_patient_id(filename):
    """
    Extract patient ID from filename using various patterns
    """
    # Remove file extension and path
    base_name = os.path.basename(filename)
    name_without_ext = os.path.splitext(base_name)[0]
    
    # Try different patterns
    patterns = [
        r'LC\d+[A-Z]?',  # LC10B, LC04B, etc.
        r'LC-\d+',        # LC-21, LC-18, etc.
        r'RT\d+[A-Z]?',   # RT01B, RT10, etc.
        r'RT-\d+',        # RT-07, RT-14, etc.
        r'W\d+',          # W2, W4, W5, etc.
        r'K\d+'           # K01
    ]
    
    for pattern in patterns:
        match = re.search(pattern, name_without_ext)
        if match:
            return match.group(0)
    
    # If no pattern matches, return None
    return None

def load_and_preprocess_csv(csv_path):
    """
    Load a single CSV file and preprocess it
    """
    df = pd.read_csv(csv_path)
    
    # Remove any non-numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df = df[numeric_cols]
    
    # Calculate mean values for each marker (feature)
    # Ensure we're getting a 1-dimensional series
    # mean_values = df.mean(axis=0)
    
    # # Convert to a dictionary to ensure 1-dimensional structure
    # mean_dict = mean_values.to_dict()
    
    # return mean_dict
    aggregated = {}
    for col in df.columns:
        aggregated[f"{col}_mean"] = df[col].mean()
        aggregated[f"{col}_std"] = df[col].std()
        aggregated[f"{col}_pct_pos"] = (df[col] > 0).mean()

    return aggregated

def create_feature_matrix(csv_dir, metadata):
    """
    Create a feature matrix where each row represents a patient and each column represents a marker
    """
    feature_data = []
    patient_ids = []
    
    # Process each CSV file
    for csv_file in glob.glob(os.path.join(csv_dir, "*_normalized.csv")):
        # Extract patient ID from filename
        #print(csv_file)
        patient_id = extract_patient_id(csv_file)
        
        if patient_id is None:
            print(f"Could not extract patient ID from: {csv_file}")
            continue
        
        # Check if this patient ID is in our metadata
        if patient_id in metadata:
            print(f"Processing {csv_file} -> {patient_id}")
            # Load and preprocess the CSV
            features = load_and_preprocess_csv(csv_file)
            feature_data.append(features)
            patient_ids.append(patient_id)
        else:
            print(f"Patient ID {patient_id} not found in metadata")
    
    # Create feature matrix from dictionaries
    feature_matrix = pd.DataFrame(feature_data, index=patient_ids)
    feature_matrix = feature_matrix.drop(
        columns=[col for col in feature_matrix.columns if col.startswith("Time_") or col.startswith("Cell_length_")],
        errors="ignore"
    )
    return feature_matrix

def perform_feature_selection(feature_matrix, metadata):
    """
    Perform feature selection using Random Forest and SHAP values
    """
    y = pd.Series([metadata[pid] for pid in feature_matrix.index])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(feature_matrix)
    X_scaled = pd.DataFrame(X_scaled, columns=feature_matrix.columns, index=feature_matrix.index)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    feature_importance = pd.DataFrame({
        'feature': feature_matrix.columns,
        'importance': model.feature_importances_
    })
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    
    # Alternative approach using SHAP values if needed
    try:
        print("Calculating SHAP values...")
        sample_size = min(50, len(X_train))
        X_sample = X_train.sample(n=sample_size, random_state=42)
        
        background = shap.sample(X_sample, 5)  
        explainer = shap.TreeExplainer(model, background)
        shap_values = explainer.shap_values(X_sample)
        
        shap_importance = np.abs(shap_values).mean(axis=0)[:, 1]
        # TODO: Decide if I want to normalize the SHAP importance. if so, uncomment the below line
        #shap_importance_normalized = shap_importance / shap_importance.sum()

        # Create a DataFrame with SHAP importance
        shap_feature_importance = pd.DataFrame({
            'feature': feature_matrix.columns,
            'shap_importance': shap_importance
        })
        shap_feature_importance = shap_feature_importance.sort_values('shap_importance', ascending=False)
        
        # Save both importance measures
        feature_importance.to_csv("feature_importance_rf.csv", index=False)
        shap_feature_importance.to_csv("feature_importance_shap.csv", index=False)
        
        return feature_importance, shap_feature_importance
        
    except Exception as e:
        print(f"Error calculating SHAP values: {e}")
        print("Using only Random Forest feature importance")
        return feature_importance, None
    
def train_and_evaluate_models(feature_matrix, metadata, top_features):
    """
    Train Logistic Regression model using cross-validation and apply KMeans for patient stratification.
    """
    print("\n=== Model Evaluation and Clustering ===")
    y = pd.Series([1 if metadata[pid] == "Responder" else 0 for pid in feature_matrix.index], index=feature_matrix.index)
    selected_features = feature_matrix[top_features]

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000)
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred_cv = cross_val_predict(model, selected_features, y, cv=cv)

    print("\n--- Cross-Validated Logistic Regression ---")
    print(confusion_matrix(y, y_pred_cv))
    print(classification_report(y, y_pred_cv))

    cv_scores = cross_val_score(model, selected_features, y, cv=cv, scoring='accuracy')
    print("Cross-Validation Accuracy Scores:", cv_scores)
    print("Mean Accuracy:", round(cv_scores.mean(), 3))

    kmeans = KMeans(n_clusters=2, random_state=42)
    cluster_labels = kmeans.fit_predict(selected_features)

    print("\n--- KMeans Clustering Results ---")
    cluster_summary = pd.DataFrame({
        'Patient_ID': selected_features.index,
        'True_Label': y.values,
        'Cluster_Label': cluster_labels
    })
    print(cluster_summary.head())

    plt.figure(figsize=(6, 4))
    sns.countplot(data=cluster_summary, x='Cluster_Label', hue='True_Label')
    plt.title("KMeans Clusters vs True Treatment Response")
    plt.xlabel("KMeans Cluster")
    plt.ylabel("Number of Patients")
    plt.legend(title="True Label")
    plt.tight_layout()
    plt.show()

    return model, kmeans, cluster_summary

def train_random_forest_model(feature_matrix, metadata, top_features):
    """
    Train a Random Forest model using cross-validation on selected features.
    """
    print("\n=== Random Forest Model ===")
    y = pd.Series([1 if metadata[pid] == "Responder" else 0 for pid in feature_matrix.index], index=feature_matrix.index)
    selected_features = feature_matrix[top_features]

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred_cv = cross_val_predict(model, selected_features, y, cv=cv)

    print("\n--- Cross-Validated Random Forest ---")
    print(confusion_matrix(y, y_pred_cv))
    print(classification_report(y, y_pred_cv))

    cv_scores = cross_val_score(model, selected_features, y, cv=cv, scoring='accuracy')
    print("Cross-Validation Accuracy Scores:", cv_scores)
    print("Mean Accuracy:", round(cv_scores.mean(), 3))

    return model

def train_lightgbm_model(feature_matrix, metadata, top_features):
    """
    Train a LightGBM model using cross-validation on selected features.
    """
    print("\n=== LightGBM Model ===")
    y = pd.Series([1 if metadata[pid] == "Responder" else 0 for pid in feature_matrix.index], index=feature_matrix.index)
    selected_features = feature_matrix[top_features]

    model = make_pipeline(
        StandardScaler(),
        lgb.LGBMClassifier(
            min_data_in_leaf=5,
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1
        )
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred_cv = cross_val_predict(model, selected_features, y, cv=cv)

    print("\n--- Cross-Validated LightGBM ---")
    print(confusion_matrix(y, y_pred_cv))
    print(classification_report(y, y_pred_cv))

    cv_scores = cross_val_score(model, selected_features, y, cv=cv, scoring='accuracy')
    print("Cross-Validation Accuracy Scores:", cv_scores)
    print("Mean Accuracy:", round(cv_scores.mean(), 3))

    return model

def cross_validated_rf_with_shap(feature_matrix, metadata, k_features=10):
    print("\n=== Strict Cross-Validated Random Forest with SHAP Feature Selection ===")

    y = pd.Series([1 if metadata[pid] == "Responder" else 0 for pid in feature_matrix.index], index=feature_matrix.index)
    X = feature_matrix.copy()
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    all_preds = []
    all_true = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        print(f"\n--- Fold {fold+1} ---")
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns, index=X_train.index)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns, index=X_test.index)

        lgb_model = LGBMClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            min_data_in_leaf=1,
            random_state=42
        )
        lgb_model.fit(X_train_scaled, y_train)

        explainer = shap.TreeExplainer(lgb_model, X_train_scaled)
        shap_values = explainer.shap_values(X_train_scaled)

        mean_shap = np.abs(shap_values).mean(axis=0)
        top_k_indices = np.argsort(mean_shap)[::-1][:k_features]
        top_k_features = X.columns[top_k_indices]
        print(f"Top features for fold {fold+1}: {top_k_features.tolist()}")

        lgb_model_final = LGBMClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            min_data_in_leaf=1,
            random_state=42
        )
        lgb_model_final.fit(X_train_scaled[top_k_features], y_train)

        y_pred = lgb_model_final.predict(X_test_scaled[top_k_features])
        all_preds.extend(y_pred)
        all_true.extend(y_test)

        acc = accuracy_score(y_test, y_pred)
        print(f"Fold Accuracy: {round(acc, 3)}")

    print("\n=== Final Cross-Validated Evaluation ===")
    print(confusion_matrix(all_true, all_preds))
    print(classification_report(all_true, all_preds))
    mean_acc = accuracy_score(all_true, all_preds)
    print(f"Mean Accuracy Across Folds: {round(mean_acc, 3)}")

def holdout_lightgbm_with_shap(feature_matrix, metadata, k_features=10, test_size=0.3):
    print("\n=== Holdout Evaluation with SHAP + LightGBM ===")
    y = pd.Series([1 if metadata[pid] == "Responder" else 0 for pid in feature_matrix.index], index=feature_matrix.index)
    X = feature_matrix.copy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns, index=X_test.index)

    # Train LightGBM on training data
    lgb_model = LGBMClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        min_data_in_leaf=1,
        random_state=42
    )
    lgb_model.fit(X_train_scaled, y_train)

    # SHAP for feature selection on training set
    explainer = shap.TreeExplainer(lgb_model, X_train_scaled)
    shap_values = explainer.shap_values(X_train_scaled)
    mean_shap = np.abs(shap_values).mean(axis=0)  

    top_k_indices = np.argsort(mean_shap)[::-1][:k_features]
    top_k_features = X.columns[top_k_indices]
    print(f"\nTop {k_features} SHAP Features: {top_k_features.tolist()}")

    lgb_model_final = LGBMClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        min_data_in_leaf=1,
        random_state=42
    )
    lgb_model_final.fit(X_train_scaled[top_k_features], y_train)

    y_pred = lgb_model_final.predict(X_test_scaled[top_k_features])

    print("\n=== Final Evaluation on Holdout Test Set ===")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print(f"Test Accuracy: {round(accuracy_score(y_test, y_pred), 3)}")

def stable_shap_holdout_lightgbm(feature_matrix, metadata, k_per_fold=10, top_n_global=10, test_size=0.2):
    print("\n=== Stable SHAP Holdout LightGBM Evaluation ===")

    # Step 1: Prepare data
    y = pd.Series([1 if metadata[pid] == "Responder" else 0 for pid in feature_matrix.index], index=feature_matrix.index)
    X = feature_matrix.copy()

    # Step 2: Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)

    # Step 3: SHAP + CV on training set to get stable features
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    feature_counts = Counter()

    for fold, (cv_train_idx, _) in enumerate(skf.split(X_train, y_train)):
        X_cv_train = X_train.iloc[cv_train_idx]
        y_cv_train = y_train.iloc[cv_train_idx]

        scaler = StandardScaler()
        X_cv_train_scaled = pd.DataFrame(scaler.fit_transform(X_cv_train), columns=X_cv_train.columns, index=X_cv_train.index)

        model = LGBMClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_data_in_leaf=1,
            random_state=fold
        )
        model.fit(X_cv_train_scaled, y_cv_train)

        explainer = shap.TreeExplainer(model, X_cv_train_scaled)
        shap_values = explainer.shap_values(X_cv_train_scaled)
        mean_shap = np.abs(shap_values).mean(axis=0)

        top_k = np.argsort(mean_shap)[::-1][:k_per_fold]
        top_features = X_cv_train.columns[top_k]
        print(f"Fold {fold+1} top features: {top_features.tolist()}")

        feature_counts.update(top_features)

    most_common_features = [f for f, _ in feature_counts.most_common(top_n_global)]
    print(f"\nMost frequent features across folds: {most_common_features}")

    # Step 4: Final model training + evaluation
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns, index=X_test.index)

    final_model = LGBMClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=4,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        min_data_in_leaf=2,
        random_state=42
    )
    final_model.fit(X_train_scaled[most_common_features], y_train)
    y_pred = final_model.predict(X_test_scaled[most_common_features])

    # Step 5: Evaluation
    print("\n=== Final Evaluation on Holdout Test Set ===")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print(f"Test Accuracy: {round(accuracy_score(y_test, y_pred), 3)}")

def main():
    # Define paths
    metadata_path = "patient_metadata.csv"
    csv_dir = "converted_csv_files"
    
    # Load metadata
    print("Loading metadata...")
    metadata = load_metadata(metadata_path)
    
    # Create feature matrix
    print("Creating feature matrix...")
    feature_matrix = create_feature_matrix(csv_dir, metadata)
    print(f"Created feature matrix with shape: {feature_matrix.shape}")
    
    # Perform feature selection
    print("Performing feature selection...")
    rf_importance, shap_importance = perform_feature_selection(feature_matrix, metadata)
    
    # Print top 10 most important features from Random Forest
    print("\nTop 10 most important features (Random Forest):")
    print(rf_importance.head(10))
    
    # Print top 10 most important features from SHAP if available
    if shap_importance is not None:
        print("\nTop 10 most important features (SHAP):")
        print(shap_importance.head(10))

        top_features = shap_importance['feature'].head(10).tolist()
        print(f"\nTraining models with top SHAP features: {top_features}")
        train_and_evaluate_models(feature_matrix, metadata, top_features)
        train_random_forest_model(feature_matrix, metadata, top_features)
        train_lightgbm_model(feature_matrix, metadata, top_features)
        cross_validated_rf_with_shap(feature_matrix, metadata, k_features=10)
        holdout_lightgbm_with_shap(feature_matrix, metadata)

        # This is the function that was used to train and test model
        # The other functions don't work, but were kept to show the process
        stable_shap_holdout_lightgbm(feature_matrix, metadata, k_per_fold=10, top_n_global=10, test_size=0.35)
    else:
        print("SHAP importance not available. Skipping model training.")

if __name__ == "__main__":
    main() 