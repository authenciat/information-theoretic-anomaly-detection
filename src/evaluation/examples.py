"""
Examples demonstrating how to use the data processing pipeline and evaluation framework.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor

from src.data.preprocessing import load_processed_data
from src.models.anomaly_detector import InfoTheoreticAnomalyDetector
from src.evaluation.metrics import (
    evaluate_anomaly_detector,
    compare_models,
    evaluate_threshold_sensitivity,
    feature_performance_analysis
)


def compare_with_baselines():
    """
    Compare the information-theoretic anomaly detector with baseline methods.
    """
    print("Loading and preprocessing data...")
    X_train, X_test, y_test, transformer = load_processed_data(
        load_unsupervised=True, 
        sample_size=100000, 
        random_state=42
    )
    
    print(f"Data shapes - X_train: {X_train.shape}, X_test: {X_test.shape}")
    
    # Initialize models
    print("Initializing models...")
    models = {
        'InfoTheoretic': InfoTheoreticAnomalyDetector(n_components=20, bandwidth=0.1),
        'IsolationForest': IsolationForest(random_state=42, contamination=0.1),
        'OneClassSVM': OneClassSVM(nu=0.1, kernel='rbf'),
        'LOF': LocalOutlierFactor(n_neighbors=20, contamination=0.1, novelty=True)
    }
    
    # Train models
    print("Training models...")
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train)
    
    # Get anomaly scores for each model
    print("Generating anomaly scores...")
    scores = {}
    
    # Information-theoretic model
    scores['InfoTheoretic'] = models['InfoTheoretic'].score_samples(X_test)['combined_score']
    
    # Isolation Forest (negative of decision_function for consistency)
    scores['IsolationForest'] = -models['IsolationForest'].decision_function(X_test)
    
    # OneClassSVM (negative of decision_function for consistency)
    scores['OneClassSVM'] = -models['OneClassSVM'].decision_function(X_test)
    
    # LOF (negative of decision_function for consistency)
    scores['LOF'] = -models['LOF'].decision_function(X_test)
    
    # Compare models
    print("Comparing models...")
    comparison_results = compare_models(y_test, scores, plot=True)
    
    print("\nModel Comparison Results:")
    print(comparison_results[['model', 'precision', 'recall', 'f1_score', 'roc_auc']])
    
    # Evaluate threshold sensitivity for the information-theoretic model
    print("\nEvaluating threshold sensitivity for the information-theoretic model...")
    threshold_results = evaluate_threshold_sensitivity(
        y_test, 
        scores['InfoTheoretic'], 
        n_thresholds=20
    )
    
    return comparison_results, threshold_results


def analyze_feature_performance():
    """
    Analyze which features are most effective for anomaly detection.
    """
    print("Loading raw data to analyze feature performance...")
    # Load the raw data first
    from src.data.preprocessing import load_kdd_data, preprocess_labels
    
    # Load a sample of the raw data
    df = load_kdd_data(sample_size=50000, random_state=42)
    
    # Preprocess labels
    df = preprocess_labels(df)
    
    # Split data directly without transforming
    from sklearn.model_selection import train_test_split
    from src.data.preprocessing import CATEGORICAL_FEATURES, NUMERIC_FEATURES
    
    # Create feature and target arrays
    X = df[CATEGORICAL_FEATURES + NUMERIC_FEATURES]
    y = df['is_anomaly']
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Analyze individual feature performance on the untransformed data
    print("\nAnalyzing individual feature performance...")
    feature_metrics = feature_performance_analysis(
        X_test, 
        y_test,
        feature_names=X_test.columns.tolist(),
        n_features=5
    )
    
    print("\nTop Features by ROC AUC:")
    print(feature_metrics.head(10))
    
    return feature_metrics


if __name__ == "__main__":
    import sys
    
    # Default to running both parts
    run_comparison = True
    run_feature_analysis = True
    
    # Parse command line arguments if provided
    if len(sys.argv) > 1:
        if sys.argv[1] == "comparison":
            run_feature_analysis = False
        elif sys.argv[1] == "features":
            run_comparison = False
    
    if run_comparison:
        print("Running comparison with baseline methods...")
        comparison_results, threshold_results = compare_with_baselines()
    
    if run_feature_analysis:
        print("\nAnalyzing feature performance...")
        feature_metrics = analyze_feature_performance()
    
    print("\nExamples completed successfully!") 