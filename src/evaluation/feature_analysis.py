"""
Feature analysis utilities for anomaly detection.

This module provides functions for analyzing and visualizing feature importance
and performance in anomaly detection tasks.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, roc_curve
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA

def feature_performance_analysis(X, y, feature_names=None, n_features=5):
    """
    Analyze how individual features perform for anomaly detection.
    
    Args:
        X: Feature matrix
        y: True labels
        feature_names: List of feature names (if None, use column indices)
        n_features: Number of top features to visualize
        
    Returns:
        DataFrame with feature performance metrics and visualization
    """
    # Ensure X is a DataFrame
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
        
    # Get feature names if not provided
    if feature_names is None:
        feature_names = X.columns.tolist()
    
    # Initialize list to store results
    feature_metrics = []
    
    # Evaluate each feature individually
    for i, feature in enumerate(feature_names):
        # Get feature values
        feature_values = X.iloc[:, i] if isinstance(feature, int) else X[feature]
        
        # Calculate ROC AUC for the feature
        try:
            auc_score = roc_auc_score(y, feature_values)
            # If AUC < 0.5, invert the feature
            if auc_score < 0.5:
                auc_score = roc_auc_score(y, -feature_values)
                invert = True
            else:
                invert = False
                
            # Calculate other metrics
            fpr, tpr, _ = roc_curve(y, feature_values if not invert else -feature_values)
            roc_auc = auc(fpr, tpr)
            
            # Add to results
            feature_metrics.append({
                'Feature': feature,
                'ROC_AUC': roc_auc,
                'Inverted': invert
            })
        except:
            # Skip features that fail (e.g., constant features)
            continue
    
    # Convert to DataFrame and sort by ROC AUC
    metrics_df = pd.DataFrame(feature_metrics)
    metrics_df = metrics_df.sort_values('ROC_AUC', ascending=False)
    
    # Get top features
    top_features = metrics_df.head(n_features)
    
    # Plot ROC curves for top features
    plt.figure(figsize=(12, 8))
    
    for _, row in top_features.iterrows():
        feature = row['Feature']
        inverted = row['Inverted']
        
        # Get feature values
        feature_idx = feature_names.index(feature) if isinstance(feature, str) else feature
        feature_values = X.iloc[:, feature_idx] if isinstance(feature, int) else X[feature]
        
        if inverted:
            feature_values = -feature_values
            
        fpr, tpr, _ = roc_curve(y, feature_values)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, label=f'{feature} (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves for Top {n_features} Features')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Plot feature distributions for normal vs anomaly
    plt.figure(figsize=(15, 10))
    
    for i, (_, row) in enumerate(top_features.iterrows()):
        feature = row['Feature']
        inverted = row['Inverted']
        
        plt.subplot(2, 3, i+1)
        
        # Get feature values
        feature_idx = feature_names.index(feature) if isinstance(feature, str) else feature
        feature_values = X.iloc[:, feature_idx] if isinstance(feature, int) else X[feature]
        
        if inverted:
            feature_values = -feature_values
            feature = f"{feature} (inverted)"
        
        # Split by class
        normal_values = feature_values[y == 0]
        anomaly_values = feature_values[y == 1]
        
        # Plot distributions
        sns.histplot(normal_values, color='blue', alpha=0.5, label='Normal', kde=True)
        sns.histplot(anomaly_values, color='red', alpha=0.5, label='Anomaly', kde=True)
        
        plt.title(f'{feature}')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.legend()
        
        # Stop after n_features
        if i >= n_features - 1:
            break
    
    plt.tight_layout()
    plt.show()
    
    return metrics_df


def visualize_anomaly_explanation(explanation, feature_names=None, n_top_features=10):
    """
    Visualize the explanation of why a specific instance is anomalous.
    
    Args:
        explanation: Output from InfoTheoreticAnomalyDetector.explain_anomaly method
        feature_names: Original feature names (if None, will use numeric indices)
        n_top_features: Number of top features to display
        
    Returns:
        None (displays visualizations)
    """
    # Extract components from the explanation
    scores = explanation['scores']
    feature_contributions = explanation['feature_contributions']
    
    # Set up the figure
    plt.figure(figsize=(18, 12))
    
    # 1. Score components breakdown - radar chart
    plt.subplot(2, 2, 1)
    
    # Prepare radar chart
    categories = list(scores.keys())
    if 'combined_score' in categories:
        categories.remove('combined_score')  # Remove combined score from radar chart
    
    # Normalize scores for radar chart
    values = [scores[cat] for cat in categories]
    values_normalized = []
    for val in values:
        if val == 0:  # Avoid division by zero
            values_normalized.append(0)
        else:
            # Log transform and normalize to [0,1]
            normalized = np.log1p(val) / max(np.log1p(max(values)), 1)
            values_normalized.append(normalized)
    
    # Compute angles for radar chart
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    values_normalized += values_normalized[:1]  # Close the loop
    
    # Draw radar chart
    ax = plt.subplot(2, 2, 1, polar=True)
    plt.polar(angles, values_normalized, marker='o', linestyle='-', linewidth=2)
    plt.fill(angles, values_normalized, alpha=0.25)
    
    # Fix axis to go clockwise and start from the top
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    # Set labels and title
    plt.xticks(angles[:-1], categories)
    plt.title("Anomaly Score Components", size=14)
    
    # 2. Score values bar chart
    plt.subplot(2, 2, 2)
    score_values = list(scores.values())
    score_names = list(scores.keys())
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(score_names)))
    
    # Check if we need logarithmic scale (if max value is 100x greater than min non-zero value)
    non_zero_values = [v for v in score_values if v > 0]
    if non_zero_values:
        min_non_zero = min(non_zero_values)
        max_value = max(score_values)
        
        use_log_scale = max_value > min_non_zero * 100
        
        # Create two subplots side by side
        plt.subplot(2, 2, 2)
        
        if use_log_scale:
            # Log scale for large value differences
            plt.bar(score_names, score_values, color=colors)
            plt.yscale('log')
            plt.title("Anomaly Score Values (Log Scale)", size=14)
        else:
            # Linear scale if values are comparable
            plt.bar(score_names, score_values, color=colors)
            plt.title("Anomaly Score Values", size=14)
        
        # Add value labels on top of bars
        for i, v in enumerate(score_values):
            if v > 0:
                # Format large values with scientific notation
                if v > 1000:
                    v_text = f"{v:.2e}"
                else:
                    v_text = f"{v:.2f}"
                plt.text(i, v, v_text, ha='center', va='bottom', rotation=90, fontsize=8)
    else:
        # Fallback if all values are zero
        plt.bar(score_names, score_values, color=colors)
        plt.title("Anomaly Score Values", size=14)
        
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # 3. Feature contributions to principal components
    plt.subplot(2, 2, 3)
    
    # Aggregate feature contributions across all PCs
    if feature_names is None:
        feature_names = [f"Feature_{i}" for i in range(len(next(iter(feature_contributions.values()))))]
    # Convert feature_names to list if it's a numpy array
    elif isinstance(feature_names, np.ndarray):
        feature_names = feature_names.tolist()
    
    total_contributions = np.zeros(len(feature_names))
    for pc_name, contributions in feature_contributions.items():
        for i, contrib in enumerate(contributions):
            total_contributions[i] += contrib
    
    # Create DataFrame for feature contributions
    contribution_df = pd.DataFrame({
        'Feature': feature_names,
        'Contribution': total_contributions
    })
    
    # Sort by absolute contribution and get top features
    contribution_df['Abs_Contribution'] = np.abs(contribution_df['Contribution'])
    contribution_df = contribution_df.sort_values('Abs_Contribution', ascending=False)
    top_contributions = contribution_df.head(n_top_features)
    
    # Plot feature contributions
    plt.barh(top_contributions['Feature'], top_contributions['Contribution'])
    plt.title("Top Feature Contributions to Anomaly", size=14)
    plt.xlabel("Contribution Score")
    plt.tight_layout()
    
    # 4. Feature contribution heatmap
    plt.subplot(2, 2, 4)
    
    # Create a matrix of feature contributions for each PC
    pc_names = list(feature_contributions.keys())
    contribution_matrix = np.array([contributions for contributions in feature_contributions.values()])
    
    # Get top features for the heatmap
    top_feature_indices = contribution_df.head(n_top_features).index
    top_feature_names = contribution_df.head(n_top_features)['Feature']
    
    # Plot heatmap
    sns.heatmap(contribution_matrix[:, top_feature_indices],
                xticklabels=top_feature_names,
                yticklabels=pc_names,
                cmap='RdBu_r',
                center=0,
                annot=True,
                fmt='.2f')
    plt.title("Feature Contributions by Principal Component", size=14)
    plt.tight_layout()
    
    plt.show() 