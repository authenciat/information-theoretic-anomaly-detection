"""
Examples demonstrating how to use the data processing pipeline and evaluation framework.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor

from src.data.preprocessing import load_processed_data, load_kdd_data, preprocess_labels
from src.models.anomaly_detector import InfoTheoreticAnomalyDetector
from src.evaluation.metrics import (
    evaluate_anomaly_detector,
    compare_models,
    evaluate_threshold_sensitivity,
    feature_performance_analysis,
    visualize_anomaly_explanation
)


def compare_with_baselines(X_train, X_test, y_test, transformer, use_adaptive=True):
    """
    Compare the information-theoretic anomaly detector with baseline methods.
    
    Args:
        X_train: Training data
        X_test: Test data
        y_test: Test labels
        transformer: Fitted transformer
        use_adaptive: Whether to use adaptive bandwidth selection
    """
    print(f"Data shapes - X_train: {X_train.shape}, X_test: {X_test.shape}")
    
    # Initialize models
    print("Initializing models...")
    
    # For the InfoTheoretic model, use adaptive bandwidth if specified
    if use_adaptive:
        print("Using adaptive bandwidth selection for InfoTheoreticAnomalyDetector")
        info_model = InfoTheoreticAnomalyDetector(
            n_components=20, 
            bandwidth=None,  # None triggers automatic bandwidth selection
            n_neighbors=20,
        )
    else:
        print("Using fixed bandwidth for InfoTheoreticAnomalyDetector")
        info_model = InfoTheoreticAnomalyDetector(
            n_components=20, 
            bandwidth=0.1,
            n_neighbors=20
        )
    
    models = {
        'InfoTheoretic': info_model,
        'IsolationForest': IsolationForest(random_state=42, contamination=0.1),
        'OneClassSVM': OneClassSVM(nu=0.1, kernel='rbf'),
        'LOF': LocalOutlierFactor(n_neighbors=20, contamination=0.1, novelty=True)
    }
    
    # Train models
    print("Training models...")
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train)
    
    # Optionally optimize weights if we have validation data
    if use_adaptive and y_test is not None:
        # Create a small validation set from the test set
        from sklearn.model_selection import train_test_split
        X_val, X_test_reduced, y_val, y_test_reduced = train_test_split(
            X_test, y_test, test_size=0.7, random_state=42, stratify=y_test
        )
        
        print("Optimizing weights for InfoTheoreticAnomalyDetector...")
        optimization_results = models['InfoTheoretic'].optimize_weights(X_val, y_val)
        print(f"Optimized weights: {optimization_results['best_weights']}")
        print(f"Validation metrics: {optimization_results['metrics']}")
        
        # Use the reduced test set to avoid data leakage
        X_test = X_test_reduced
        y_test = y_test_reduced
    
    # Get anomaly scores for each model
    print("Generating anomaly scores...")
    scores = {}
    
    # Information-theoretic model (use adaptive local bandwidth if specified)
    scores['InfoTheoretic'] = models['InfoTheoretic'].score_samples(
        X_test, use_adaptive_local=use_adaptive
    )['combined_score']
    
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


def analyze_feature_performance(df):
    """
    Analyze which features are most effective for anomaly detection.
    
    Args:
        df: Raw DataFrame containing the data
    """
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


def explain_anomaly_instance(X_train, X_test, y_test, transformer, use_adaptive=True):
    """
    Demonstrate how to explain why a specific instance was flagged as anomalous.
    
    Args:
        X_train: Training data
        X_test: Test data
        y_test: Test labels
        transformer: Fitted transformer
        use_adaptive: Whether to use adaptive bandwidth selection
    """
    print(f"Data shapes - X_train: {X_train.shape}, X_test: {X_test.shape}")
    
    # Initialize and train the InfoTheoretic model
    print("Initializing model...")
    if use_adaptive:
        print("Using adaptive bandwidth selection")
        detector = InfoTheoreticAnomalyDetector(
            n_components=20, 
            bandwidth=None,  # None triggers automatic bandwidth selection
            n_neighbors=20
        )
    else:
        detector = InfoTheoreticAnomalyDetector(
            n_components=20, 
            bandwidth=0.1,
            n_neighbors=20
        )
    
    print("Training model...")
    detector.fit(X_train)
    
    # Score samples to find anomalies
    print("Scoring samples...")
    scores = detector.score_samples(X_test, use_adaptive_local=use_adaptive)
    combined_scores = scores['combined_score']
    
    # Find an actual anomaly with high score
    if y_test is not None:
        # Find the anomaly with the highest score
        anomaly_indices = np.where(y_test == 1)[0]
        if len(anomaly_indices) > 0:
            # Get scores for anomalies only
            anomaly_scores = combined_scores[anomaly_indices]
            # Find the index of the highest scoring anomaly
            highest_anomaly_idx = anomaly_indices[np.argmax(anomaly_scores)]
            instance_idx = highest_anomaly_idx
            print(f"Explaining a true anomaly (index {instance_idx}) with score {combined_scores[instance_idx]:.4f}")
        else:
            # If no true anomalies, just use the highest scoring instance
            instance_idx = np.argmax(combined_scores)
            print(f"No true anomalies in test set. Explaining highest scoring instance (index {instance_idx}) with score {combined_scores[instance_idx]:.4f}")
    else:
        # If we don't have labels, just use the highest scoring instance
        instance_idx = np.argmax(combined_scores)
        print(f"Explaining highest scoring instance (index {instance_idx}) with score {combined_scores[instance_idx]:.4f}")
    
    # Get feature names from the transformer if available
    feature_names = None
    if hasattr(transformer, 'feature_names_out_'):
        feature_names = transformer.feature_names_out_
    
    # Generate explanation for the selected instance
    print("Generating explanation...")
    explanation = detector.explain_anomaly(X_test, instance_idx)
    
    # Visualize the explanation
    print("Visualizing explanation...")
    visualize_anomaly_explanation(explanation, feature_names=feature_names)
    
    return explanation, instance_idx


if __name__ == "__main__":
    import sys
    
    # Default to running all parts
    run_comparison = True
    run_feature_analysis = True
    run_explanation = True
    use_adaptive = True
    
    # Parse command line arguments if provided
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            if arg == "comparison":
                run_feature_analysis = False
                run_explanation = False
            elif arg == "features":
                run_comparison = False
                run_explanation = False
            elif arg == "explain":
                run_comparison = False
                run_feature_analysis = False
            elif arg == "noadaptive":
                use_adaptive = False
    
    # Load data once
    print("Loading and preprocessing data...")
    X_train, X_test, y_test, transformer = load_processed_data(
        load_unsupervised=True, 
        sample_size=100000, 
        random_state=42
    )
    
    # Load raw data for feature analysis if needed
    if run_feature_analysis:
        print("\nLoading raw data for feature analysis...")
        df_raw = load_kdd_data(sample_size=50000, random_state=42)
    
    if run_comparison:
        print("Running comparison with baseline methods...")
        comparison_results, threshold_results = compare_with_baselines(
            X_train, X_test, y_test, transformer, use_adaptive=use_adaptive
        )
    
    if run_feature_analysis:
        print("\nAnalyzing feature performance...")
        feature_metrics = analyze_feature_performance(df_raw)
    
    if run_explanation:
        print("\nExplaining an anomaly instance...")
        explanation, instance_idx = explain_anomaly_instance(
            X_train, X_test, y_test, transformer, use_adaptive=use_adaptive
        )
    
    print("\nExamples completed successfully!") 