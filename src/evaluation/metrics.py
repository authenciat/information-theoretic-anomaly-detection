"""
Evaluation metrics for anomaly detection.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_recall_curve, 
    roc_curve, 
    auc, 
    precision_score, 
    recall_score, 
    f1_score, 
    accuracy_score, 
    confusion_matrix, 
    average_precision_score,
    roc_auc_score
)

# Import feature analysis functions
from .feature_analysis import feature_performance_analysis, visualize_anomaly_explanation


def evaluate_anomaly_detector(y_true, y_pred_scores, threshold=None, plot=True):
    """
    Evaluate an anomaly detector using multiple metrics.
    
    Args:
        y_true: True binary labels (0=normal, 1=anomaly)
        y_pred_scores: Predicted anomaly scores (higher means more anomalous)
        threshold: Classification threshold for scores. If None, use ROC analysis to find optimal.
        plot: Whether to generate and display evaluation plots
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Ensure inputs are numpy arrays
    y_true = np.asarray(y_true)
    y_pred_scores = np.asarray(y_pred_scores)
    
    # Compute ROC curve and ROC AUC
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_pred_scores)
    roc_auc = auc(fpr, tpr)
    
    # Compute Precision-Recall curve and PR AUC
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_pred_scores)
    pr_auc = auc(recall, precision)
    average_precision = average_precision_score(y_true, y_pred_scores)
    
    # Determine threshold if not provided
    if threshold is None:
        # Find threshold that maximizes F1 score
        f1_scores = np.zeros_like(pr_thresholds)
        for i, threshold in enumerate(pr_thresholds):
            y_pred = (y_pred_scores >= threshold).astype(int)
            f1_scores[i] = f1_score(y_true, y_pred)
        
        best_idx = np.argmax(f1_scores)
        threshold = pr_thresholds[best_idx]
    
    # Compute binary predictions using the threshold
    y_pred = (y_pred_scores >= threshold).astype(int)
    
    # Compute classification metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision_at_threshold = precision_score(y_true, y_pred)
    recall_at_threshold = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Calculate additional metrics
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    fpr_at_threshold = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    # Plot evaluation curves if requested
    if plot:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # ROC curve
        ax1.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax1.plot([0, 1], [0, 1], 'k--')
        ax1.set_xlim([0.0, 1.0])
        ax1.set_ylim([0.0, 1.05])
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('Receiver Operating Characteristic')
        ax1.legend(loc="lower right")
        
        # Precision-Recall curve
        ax2.plot(recall, precision, label=f'PR curve (AUC = {pr_auc:.3f})')
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title('Precision-Recall Curve')
        ax2.legend(loc="lower left")
        
        plt.tight_layout()
        plt.show()
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Normal', 'Anomaly'], 
                    yticklabels=['Normal', 'Anomaly'])
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix')
        plt.show()
    
    # Return all metrics
    return {
        'threshold': threshold,
        'accuracy': accuracy,
        'precision': precision_at_threshold,
        'recall': recall_at_threshold,
        'f1_score': f1,
        'specificity': specificity,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'average_precision': average_precision,
        'fpr_at_threshold': fpr_at_threshold,
        'true_positives': tp,
        'false_positives': fp,
        'true_negatives': tn,
        'false_negatives': fn
    }


def compare_models(y_true, score_dict, thresholds=None, plot=True):
    """
    Compare multiple anomaly detection models on the same dataset.
    
    Args:
        y_true: True binary labels (0=normal, 1=anomaly)
        score_dict: Dictionary of model name to anomaly scores
        thresholds: Dictionary of model name to threshold (if None, find optimal for each)
        plot: Whether to plot comparison curves
        
    Returns:
        DataFrame with comparison metrics for each model
    """
    results = []
    
    # Set up threshold dictionary if not provided
    if thresholds is None:
        thresholds = {model_name: None for model_name in score_dict.keys()}
    
    # Evaluate each model
    for model_name, scores in score_dict.items():
        threshold = thresholds.get(model_name, None)
        model_metrics = evaluate_anomaly_detector(y_true, scores, threshold, plot=False)
        model_metrics['model'] = model_name
        results.append(model_metrics)
    
    # Convert to DataFrame and sort by F1 score
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('f1_score', ascending=False)
    
    # Plot comparison if requested
    if plot:
        # Set up the matplotlib figure
        plt.figure(figsize=(15, 10))
        
        # ROC curves
        plt.subplot(2, 2, 1)
        for model_name, scores in score_dict.items():
            fpr, tpr, _ = roc_curve(y_true, scores)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend(loc="lower right")
        
        # Precision-Recall curves
        plt.subplot(2, 2, 2)
        for model_name, scores in score_dict.items():
            precision, recall, _ = precision_recall_curve(y_true, scores)
            pr_auc = auc(recall, precision)
            plt.plot(recall, precision, label=f'{model_name} (AUC = {pr_auc:.3f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves')
        plt.legend(loc="lower left")
        
        # F1 scores
        plt.subplot(2, 2, 3)
        models = results_df['model'].tolist()
        f1_scores = results_df['f1_score'].tolist()
        colors = sns.color_palette('husl', len(models))
        
        bars = plt.bar(models, f1_scores, color=colors)
        plt.ylim([0, 1.0])
        plt.xlabel('Model')
        plt.ylabel('F1 Score')
        plt.title('F1 Scores Comparison')
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', rotation=0)
        
        # Precision & Recall
        plt.subplot(2, 2, 4)
        width = 0.35
        x = np.arange(len(models))
        
        precision_values = results_df['precision'].tolist()
        recall_values = results_df['recall'].tolist()
        
        bars1 = plt.bar(x - width/2, precision_values, width, label='Precision', color='skyblue')
        bars2 = plt.bar(x + width/2, recall_values, width, label='Recall', color='lightcoral')
        
        plt.ylim([0, 1.0])
        plt.xlabel('Model')
        plt.ylabel('Score')
        plt.title('Precision & Recall Comparison')
        plt.xticks(x, models, rotation=45, ha='right')
        plt.legend()
        
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom', rotation=0)
        
        plt.tight_layout()
        plt.show()
    
    return results_df


def evaluate_threshold_sensitivity(y_true, y_pred_scores, thresholds=None, n_thresholds=10):
    """
    Evaluate how performance metrics change with different thresholds.
    
    Args:
        y_true: True binary labels (0=normal, 1=anomaly)
        y_pred_scores: Predicted anomaly scores
        thresholds: List of thresholds to evaluate (if None, generate n_thresholds evenly spaced)
        n_thresholds: Number of thresholds to generate if not provided
        
    Returns:
        DataFrame with metrics at each threshold
    """
    # Generate thresholds if not provided
    if thresholds is None:
        # Get the range of scores
        min_score = np.min(y_pred_scores)
        max_score = np.max(y_pred_scores)
        thresholds = np.linspace(min_score, max_score, n_thresholds)
    
    # Initialize lists to store results
    results = []
    
    # Compute metrics for each threshold
    for threshold in thresholds:
        y_pred = (y_pred_scores >= threshold).astype(int)
        
        # Skip thresholds that result in all one class
        if len(np.unique(y_pred)) < 2:
            continue
        
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)
        
        # Compute confusion matrix values
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        results.append({
            'threshold': threshold,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'specificity': specificity,
            'fpr': fpr
        })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Plot metrics vs threshold
    plt.figure(figsize=(15, 10))
    
    # F1, Precision, Recall plot
    plt.subplot(2, 2, 1)
    plt.plot(results_df['threshold'], results_df['f1_score'], label='F1', marker='o')
    plt.plot(results_df['threshold'], results_df['precision'], label='Precision', marker='s')
    plt.plot(results_df['threshold'], results_df['recall'], label='Recall', marker='^')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('F1, Precision, and Recall vs. Threshold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Accuracy & Specificity plot
    plt.subplot(2, 2, 2)
    plt.plot(results_df['threshold'], results_df['accuracy'], label='Accuracy', marker='o')
    plt.plot(results_df['threshold'], results_df['specificity'], label='Specificity', marker='s')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Accuracy and Specificity vs. Threshold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Find the threshold with the best F1 score
    best_f1_idx = results_df['f1_score'].idxmax()
    best_threshold = results_df.loc[best_f1_idx, 'threshold']
    best_f1 = results_df.loc[best_f1_idx, 'f1_score']
    
    # F1 score vs threshold with best point highlighted
    plt.subplot(2, 2, 3)
    plt.plot(results_df['threshold'], results_df['f1_score'], marker='o')
    plt.scatter([best_threshold], [best_f1], color='red', s=100, zorder=5)
    plt.annotate(f'Best F1: {best_f1:.3f} at {best_threshold:.3f}',
                xy=(best_threshold, best_f1), xytext=(best_threshold, best_f1 - 0.1),
                arrowprops=dict(arrowstyle='->', color='red'))
    plt.xlabel('Threshold')
    plt.ylabel('F1 Score')
    plt.title('F1 Score vs. Threshold with Best Point')
    plt.grid(True, alpha=0.3)
    
    # Precision vs Recall for different thresholds
    plt.subplot(2, 2, 4)
    plt.plot(results_df['recall'], results_df['precision'], '-o')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision vs. Recall at Different Thresholds')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return results_df 