# Information-theoretic anomaly detection models 
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity, NearestNeighbors
from scipy.stats import entropy
from scipy.special import logsumexp
from sklearn.model_selection import GridSearchCV

class InfoTheoreticAnomalyDetector:
    def __init__(self, n_components=10, bandwidth=None, bandwidth_range=None, n_neighbors=10, weights=None):
        self.n_components = n_components
        self.bandwidth = bandwidth
        self.bandwidth_range = bandwidth_range or np.logspace(-2, 1, 20)  # Default range for bandwidth search
        self.n_neighbors = n_neighbors
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components)
        self.kde = None  # Will be initialized during fit
        self.nn = NearestNeighbors(n_neighbors=n_neighbors)
        
        # Default weights for combining scores if not provided
        if weights is None:
            # Equal weights by default
            self.weights = [0.25, 0.25, 0.25, 0.25]
        else:
            self.weights = weights
    
    def _select_optimal_bandwidth(self, X):
        """
        Select optimal bandwidth for KDE using cross-validation.
        
        Args:
            X: Input data (already dimensionality reduced)
            
        Returns:
            Optimal bandwidth value
        """
        # Create a grid of bandwidth parameters to search
        grid_params = {'bandwidth': self.bandwidth_range}
        
        # Create the grid search object
        grid = GridSearchCV(
            KernelDensity(),
            grid_params,
            cv=min(5, len(X) // 5),  # Use 5-fold CV or less if data is small
            n_jobs=-1  # Use all available processors
        )
        
        # Fit the grid search
        grid.fit(X)
        
        # Return the optimal bandwidth
        return grid.best_params_['bandwidth']
    
    def fit(self, X):
        # Standardize data
        X_scaled = self.scaler.fit_transform(X)
        
        # Reduce dimensionality
        X_reduced = self.pca.fit_transform(X_scaled)
        
        # Store the transformed training data for local density estimation
        self.X_train_reduced = X_reduced
        
        # Select optimal bandwidth if not specified
        if self.bandwidth is None:
            self.bandwidth = self._select_optimal_bandwidth(X_reduced)
            print(f"Selected optimal bandwidth: {self.bandwidth:.6f}")
        
        # Fit KDE for global probability estimation with optimal bandwidth
        self.kde = KernelDensity(bandwidth=self.bandwidth)
        self.kde.fit(X_reduced)
        
        # Fit nearest neighbors model for local density estimation
        self.nn.fit(X_reduced)
        
        return self
    
    def compute_local_kde(self, x, neighbors_idx, adaptive_local=True):
        """
        Compute local KDE around a point using its neighbors
        
        Args:
            x: The point to compute local KDE for
            neighbors_idx: Indices of neighbors in training set
            adaptive_local: Whether to use adaptive bandwidth for local KDE
            
        Returns:
            Fitted local KDE model
        """
        if not adaptive_local or len(neighbors_idx) < 20:
            # Use global bandwidth if not enough neighbors or adaptive not requested
            local_kde = KernelDensity(bandwidth=self.bandwidth)
        else:
            # Calculate local bandwidth based on neighbor distances
            neighbors_data = self.X_train_reduced[neighbors_idx]
            
            # Use Silverman's rule of thumb for local bandwidth
            # h = 0.9 * min(std, IQR/1.34) * n^(-1/5)
            n = len(neighbors_data)
            d = neighbors_data.shape[1]  # dimensionality
            
            # Calculate standard deviation for each dimension
            std_dev = np.std(neighbors_data, axis=0)
            
            # Calculate IQR for each dimension
            q75, q25 = np.percentile(neighbors_data, [75, 25], axis=0)
            iqr = q75 - q25
            iqr_normalized = iqr / 1.34
            
            # Choose the smaller of std and normalized IQR for each dimension
            min_stats = np.minimum(std_dev, iqr_normalized)
            
            # Calculate Silverman's bandwidth
            local_bandwidth = 0.9 * np.mean(min_stats) * n**(-1/(4+d))
            
            # Ensure bandwidth is not too small or too large
            local_bandwidth = max(local_bandwidth, self.bandwidth * 0.1)
            local_bandwidth = min(local_bandwidth, self.bandwidth * 10)
            
            local_kde = KernelDensity(bandwidth=local_bandwidth)
        
        # Fit the local KDE
        local_kde.fit(self.X_train_reduced[neighbors_idx])
        return local_kde
    
    def estimate_differential_entropy(self, density_values):
        """Estimate differential entropy using Monte Carlo approximation"""
        # Differential entropy is -E[log(p(x))]
        # Use Monte Carlo approximation: -mean(log(p(x)))
        # Density values are already in log space from KDE
        return -np.mean(density_values)
    
    def score_samples(self, X, use_adaptive_local=True):
        """
        Compute multiple information-theoretic anomaly scores for each sample.
        
        Args:
            X: Input data
            use_adaptive_local: Whether to use adaptive local bandwidth
            
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
            # Create a local KDE model using the neighbors with optional adaptive bandwidth
            local_kde = self.compute_local_kde(X_reduced[i], neighbor_indices, adaptive_local=use_adaptive_local)
            
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
    
    def predict(self, X, threshold=None, percentile=95, use_adaptive_local=True):
        """
        Predict anomalies based on combined score.
        
        Args:
            X: Input data
            threshold: Score threshold for anomaly detection
            percentile: If threshold is None, use this percentile to determine threshold
            use_adaptive_local: Whether to use adaptive local bandwidth
            
        Returns:
            Binary array where 1 indicates anomaly, 0 indicates normal
        """
        scores = self.score_samples(X, use_adaptive_local=use_adaptive_local)
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
    
    def optimize_weights(self, X_val, y_val, method='grid_search', n_iter=10):
        """
        Optimize the weights of different scoring components based on validation data.
        
        Args:
            X_val: Validation features
            y_val: Validation labels (1 for anomalies, 0 for normal)
            method: Optimization method ('grid_search' or 'random')
            n_iter: Number of iterations for optimization
            
        Returns:
            Dictionary with best weights and corresponding F1 score
        """
        from sklearn.metrics import f1_score, precision_score, recall_score
        
        best_f1 = 0
        best_weights = self.weights.copy()
        best_metrics = {}
        
        # Get all score components for the validation set
        all_scores = self.score_samples(X_val)
        
        # Extract individual normalized score components
        n_samples = len(X_val)
        
        # Normalize scores to [0, 1] range for proper weighting
        def normalize(scores):
            min_val = np.min(scores)
            max_val = np.max(scores)
            if max_val > min_val:
                return (scores - min_val) / (max_val - min_val)
            return np.zeros_like(scores)
        
        norm_basic_scores = normalize(all_scores['basic_score'])
        norm_local_entropy_scores = normalize(all_scores['local_entropy_score'])
        norm_relative_entropy_scores = normalize(all_scores['relative_entropy_score'])
        norm_differential_entropy_scores = normalize(all_scores['differential_entropy_score'])
        
        if method == 'grid_search':
            # Define a grid of weight combinations to try
            # The weights should sum to 1
            weight_grid = []
            for w1 in np.linspace(0.1, 0.7, 4):  # basic score weight
                for w2 in np.linspace(0.1, 0.4, 3):  # local entropy weight
                    for w3 in np.linspace(0.1, 0.4, 3):  # relative entropy weight
                        w4 = 1.0 - (w1 + w2 + w3)  # differential entropy weight
                        if 0.05 <= w4 <= 0.4:  # ensure w4 is reasonable
                            weight_grid.append([w1, w2, w3, w4])
        else:  # random search
            # Generate random weight combinations
            weight_grid = []
            for _ in range(n_iter):
                # Generate random weights that sum to 1
                weights = np.random.dirichlet(np.ones(4))
                weight_grid.append(weights)
        
        # Try different weight combinations
        for weights in weight_grid:
            # Compute combined score with current weights
            combined_scores = (
                weights[0] * norm_basic_scores +
                weights[1] * norm_local_entropy_scores +
                weights[2] * norm_relative_entropy_scores +
                weights[3] * norm_differential_entropy_scores
            )
            
            # Optimize threshold for best F1 score
            best_threshold = None
            best_local_f1 = 0
            
            for percentile in range(80, 99, 2):
                threshold = np.percentile(combined_scores, percentile)
                y_pred = (combined_scores > threshold).astype(int)
                f1 = f1_score(y_val, y_pred)
                
                if f1 > best_local_f1:
                    best_local_f1 = f1
                    best_threshold = threshold
            
            # Final evaluation with best threshold
            y_pred = (combined_scores > best_threshold).astype(int)
            current_f1 = f1_score(y_val, y_pred)
            precision = precision_score(y_val, y_pred)
            recall = recall_score(y_val, y_pred)
            
            if current_f1 > best_f1:
                best_f1 = current_f1
                best_weights = weights
                best_metrics = {
                    'f1_score': current_f1,
                    'precision': precision,
                    'recall': recall,
                    'threshold': best_threshold
                }
        
        # Update model weights with the best ones found
        self.weights = best_weights
        
        return {
            'best_weights': best_weights,
            'metrics': best_metrics
        }

# Usage example
if __name__ == "__main__":
    # This is an example of how to use the refined InfoTheoreticAnomalyDetector
    # with adaptive bandwidth selection
    
    # Import the required libraries
    import numpy as np
    from sklearn.datasets import make_blobs
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    
    # Create a synthetic dataset with anomalies
    # Normal data: cluster with 1000 samples
    # Anomalies: scattered points (100 samples)
    X_normal, _ = make_blobs(n_samples=1000, centers=1, random_state=42)
    X_anomaly, _ = make_blobs(n_samples=100, centers=5, cluster_std=3.0, random_state=42)
    
    # Combine the data
    X = np.vstack([X_normal, X_anomaly])
    y = np.zeros(1100)
    y[1000:] = 1  # Anomalies have label 1
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Create a validation set for weight optimization
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.3, random_state=42, stratify=y_train
    )
    
    # Create the detector with automatic bandwidth selection
    detector = InfoTheoreticAnomalyDetector(
        n_components=2,  # Using 2D for this example
        bandwidth=None,  # Use automatic bandwidth selection
        n_neighbors=20
    )
    
    # Fit the detector (only on normal data for unsupervised learning)
    X_train_normal = X_train[y_train == 0]
    detector.fit(X_train_normal)
    
    # Optimize the weights using the validation set
    optimization_results = detector.optimize_weights(X_val, y_val)
    print("Optimized weights:", optimization_results['best_weights'])
    print("Validation metrics:", optimization_results['metrics'])
    
    # Make predictions on the test set
    y_pred = detector.predict(X_test, use_adaptive_local=True)
    
    # Evaluate the model
    print("\nTest Set Performance:")
    print(classification_report(y_test, y_pred))
    
    # Compare with non-adaptive bandwidth
    print("\nComparing with non-adaptive bandwidth:")
    y_pred_non_adaptive = detector.predict(X_test, use_adaptive_local=False)
    print(classification_report(y_test, y_pred_non_adaptive))