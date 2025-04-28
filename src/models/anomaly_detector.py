# Information-theoretic anomaly detection models 
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity, NearestNeighbors
from scipy.stats import entropy
from scipy.special import logsumexp

class InfoTheoreticAnomalyDetector:
    def __init__(self, n_components=10, bandwidth=0.1, n_neighbors=10, weights=None):
        self.n_components = n_components
        self.bandwidth = bandwidth
        self.n_neighbors = n_neighbors
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components)
        self.kde = KernelDensity(bandwidth=bandwidth)
        self.nn = NearestNeighbors(n_neighbors=n_neighbors)
        
        # Default weights for combining scores if not provided
        if weights is None:
            # Equal weights by default
            self.weights = [0.25, 0.25, 0.25, 0.25]
        else:
            self.weights = weights
    
    def fit(self, X):
        # Standardize data
        X_scaled = self.scaler.fit_transform(X)
        
        # Reduce dimensionality
        X_reduced = self.pca.fit_transform(X_scaled)
        
        # Store the transformed training data for local density estimation
        self.X_train_reduced = X_reduced
        
        # Fit KDE for global probability estimation
        self.kde.fit(X_reduced)
        
        # Fit nearest neighbors model for local density estimation
        self.nn.fit(X_reduced)
        
        return self
    
    def compute_local_kde(self, x, neighbors_idx):
        """Compute local KDE around a point using its neighbors"""
        # Use neighbors to create a local KDE
        local_kde = KernelDensity(bandwidth=self.bandwidth)
        local_kde.fit(self.X_train_reduced[neighbors_idx])
        return local_kde
    
    def estimate_differential_entropy(self, density_values):
        """Estimate differential entropy using Monte Carlo approximation"""
        # Differential entropy is -E[log(p(x))]
        # Use Monte Carlo approximation: -mean(log(p(x)))
        # Density values are already in log space from KDE
        return -np.mean(density_values)
    
    def score_samples(self, X):
        """
        Compute multiple information-theoretic anomaly scores for each sample.
        
        Returns:
            Dictionary of scores and combined anomaly score
        """
        # Transform data
        X_scaled = self.scaler.transform(X)
        X_reduced = self.pca.transform(X_scaled)
        
        n_samples = X_reduced.shape[0]
        
        # 1. Basic score (negative log probability under global distribution)
        basic_scores = -self.kde.score_samples(X_reduced)
        
        # Initialize arrays for other scores
        local_entropy_scores = np.zeros(n_samples)
        relative_entropy_scores = np.zeros(n_samples)
        differential_entropy_scores = np.zeros(n_samples)
        
        # Find neighbors for each sample in the reduced space
        distances, neighbors = self.nn.kneighbors(X_reduced)
        
        # Global density estimates in log space
        global_log_density = self.kde.score_samples(self.X_train_reduced)
        
        # Process each sample
        for i in range(n_samples):
            # Get indices of nearest neighbors in training set
            neighbor_indices = neighbors[i]
            
            # 2. Localized Entropy Score
            # Create a local KDE model using the neighbors
            local_kde = self.compute_local_kde(X_reduced[i], neighbor_indices)
            
            # Score the sample with local KDE
            local_log_density = local_kde.score_samples(X_reduced[i].reshape(1, -1))[0]
            
            # Score neighbors with local KDE
            neighbor_log_densities = local_kde.score_samples(self.X_train_reduced[neighbor_indices])
            
            # Convert log densities to probabilities and normalize
            neighbor_probs = np.exp(neighbor_log_densities)
            neighbor_probs = neighbor_probs / np.sum(neighbor_probs)
            
            # Compute entropy of neighbor probability distribution
            # Add small epsilon to avoid log(0)
            eps = 1e-10
            local_entropy = entropy(neighbor_probs + eps)
            local_entropy_scores[i] = local_entropy
            
            # 3. Relative Entropy Score (KL Divergence)
            # Compare local and global distributions over neighbors
            global_neighbor_log_densities = global_log_density[neighbor_indices]
            global_neighbor_probs = np.exp(global_neighbor_log_densities)
            global_neighbor_probs = global_neighbor_probs / np.sum(global_neighbor_probs)
            
            # KL(local || global) = sum(p_local * log(p_local / p_global))
            kl_divergence = entropy(neighbor_probs + eps, global_neighbor_probs + eps)
            relative_entropy_scores[i] = kl_divergence
            
            # 4. Differential Entropy Score
            # Higher values indicate higher uncertainty in local region
            # We use the entropy of the local KDE as an approximation
            differential_entropy = self.estimate_differential_entropy(neighbor_log_densities)
            differential_entropy_scores[i] = differential_entropy
        
        # Normalize scores to [0, 1] range for proper weighting
        def normalize(scores):
            min_val = np.min(scores)
            max_val = np.max(scores)
            if max_val > min_val:
                return (scores - min_val) / (max_val - min_val)
            return np.zeros_like(scores)
        
        norm_basic_scores = normalize(basic_scores)
        norm_local_entropy_scores = normalize(local_entropy_scores)
        norm_relative_entropy_scores = normalize(relative_entropy_scores)
        norm_differential_entropy_scores = normalize(differential_entropy_scores)
        
        # 5. Combined Score
        combined_scores = (
            self.weights[0] * norm_basic_scores +
            self.weights[1] * norm_local_entropy_scores +
            self.weights[2] * norm_relative_entropy_scores +
            self.weights[3] * norm_differential_entropy_scores
        )
        
        # Return all scores for potential analysis and visualization
        return {
            'basic_score': basic_scores,
            'local_entropy_score': local_entropy_scores,
            'relative_entropy_score': relative_entropy_scores,
            'differential_entropy_score': differential_entropy_scores,
            'combined_score': combined_scores
        }
    
    def predict(self, X, threshold=None, percentile=95):
        """
        Predict anomalies based on combined score.
        
        Args:
            X: Input data
            threshold: Score threshold for anomaly detection
            percentile: If threshold is None, use this percentile to determine threshold
            
        Returns:
            Binary array where 1 indicates anomaly, 0 indicates normal
        """
        scores = self.score_samples(X)
        combined_scores = scores['combined_score']
        
        if threshold is None:
            # Set threshold at the specified percentile of scores
            threshold = np.percentile(combined_scores, percentile)
        
        return (combined_scores > threshold).astype(int)
    
    def explain_anomaly(self, X, index):
        """
        Explain why a particular sample is anomalous by breaking down
        its anomaly score into components.
        
        Args:
            X: Input data
            index: Index of the sample to explain
            
        Returns:
            Dictionary with score breakdown and feature contributions
        """
        scores = self.score_samples(X)
        
        # Get all scores for the specific sample
        sample_scores = {k: v[index] for k, v in scores.items()}
        
        # Transform data to analyze feature contributions
        X_scaled = self.scaler.transform(X)
        X_reduced = self.pca.transform(X_scaled)
        
        # Get PCA components to analyze feature contributions
        components = self.pca.components_
        
        # Calculate feature contributions
        feature_contributions = {}
        for i, component in enumerate(components):
            # This shows how much each original feature contributes to the
            # principal components that are most anomalous
            contribution = np.abs(component)
            feature_contributions[f'PC{i+1}'] = contribution
        
        return {
            'scores': sample_scores,
            'feature_contributions': feature_contributions,
        }