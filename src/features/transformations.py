# Feature engineering and transformation 
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder
from sklearn.decomposition import PCA, KernelPCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel, mutual_info_classif
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import scipy.stats as stats

class AnomalyFeatureTransformer:
    """
    A comprehensive feature transformation class for anomaly detection
    that handles categorical encoding, normalization, and dimensionality reduction.
    """
    
    def __init__(self, 
                 categorical_features=None,
                 numeric_features=None,
                 scaling_method='standard',
                 categorical_encoding='onehot',
                 dimensionality_reduction=None,
                 n_components=None,
                 random_state=42):
        """
        Initialize the transformer.
        
        Args:
            categorical_features: List of categorical column names/indices
            numeric_features: List of numerical column names/indices
            scaling_method: Method to scale numerical features ('standard', 'robust')
            categorical_encoding: Method to encode categorical features ('onehot')
            dimensionality_reduction: Method for reducing dimensions (None, 'pca', 'kpca')
            n_components: Number of components for dimensionality reduction
            random_state: Random seed for reproducibility
        """
        self.categorical_features = categorical_features
        self.numeric_features = numeric_features
        self.scaling_method = scaling_method
        self.categorical_encoding = categorical_encoding
        self.dimensionality_reduction = dimensionality_reduction
        self.n_components = n_components
        self.random_state = random_state
        self.transformer = None
        self.dim_reducer = None
        self.feature_names_out_ = None
        
    def _get_scaler(self):
        """Return the appropriate scaler based on the specified method"""
        if self.scaling_method == 'standard':
            return StandardScaler()
        elif self.scaling_method == 'robust':
            return RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {self.scaling_method}")
    
    def _get_encoder(self):
        """Return the appropriate encoder based on the specified method"""
        if self.categorical_encoding == 'onehot':
            return OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        else:
            raise ValueError(f"Unknown categorical encoding method: {self.categorical_encoding}")
    
    def _get_dim_reducer(self):
        """Return the appropriate dimensionality reduction technique"""
        if self.dimensionality_reduction is None:
            return None
        
        n_components = self.n_components or 10  # Default to 10 components if not specified
        
        if self.dimensionality_reduction == 'pca':
            return PCA(n_components=n_components, random_state=self.random_state)
        elif self.dimensionality_reduction == 'kpca':
            return KernelPCA(n_components=n_components, kernel='rbf', random_state=self.random_state)
        else:
            raise ValueError(f"Unknown dimensionality reduction technique: {self.dimensionality_reduction}")
    
    def fit(self, X, y=None):
        """
        Fit the transformer to the data
        
        Args:
            X: Input features (pandas DataFrame or numpy array)
            y: Optional target for supervised dimensionality reduction
            
        Returns:
            self
        """
        # Convert to pandas DataFrame if numpy array
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        
        # Auto-detect features if not provided
        if self.categorical_features is None and self.numeric_features is None:
            self.categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
            self.numeric_features = X.select_dtypes(include=np.number).columns.tolist()
        
        # Create the column transformer
        transformers = []
        
        if self.numeric_features:
            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', self._get_scaler())
            ])
            transformers.append(('num', numeric_transformer, self.numeric_features))
        
        if self.categorical_features:
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', self._get_encoder())
            ])
            transformers.append(('cat', categorical_transformer, self.categorical_features))
        
        self.transformer = ColumnTransformer(transformers=transformers)
        
        # Fit the column transformer
        X_transformed = self.transformer.fit_transform(X)
        
        # Apply dimensionality reduction if specified
        if self.dimensionality_reduction:
            self.dim_reducer = self._get_dim_reducer()
            self.dim_reducer.fit(X_transformed, y)
        
        # Track feature names for interpretability
        if hasattr(self.transformer, 'get_feature_names_out'):
            self.feature_names_out_ = self.transformer.get_feature_names_out()
        
        return self
    
    def transform(self, X):
        """
        Transform the data
        
        Args:
            X: Input features
            
        Returns:
            Transformed features
        """
        # Convert to pandas DataFrame if numpy array
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        
        # Apply the column transformer
        X_transformed = self.transformer.transform(X)
        
        # Apply dimensionality reduction if specified
        if self.dimensionality_reduction and self.dim_reducer:
            X_transformed = self.dim_reducer.transform(X_transformed)
        
        return X_transformed
    
    def fit_transform(self, X, y=None):
        """
        Fit and transform the data
        
        Args:
            X: Input features
            y: Optional target for supervised dimensionality reduction
            
        Returns:
            Transformed features
        """
        self.fit(X, y)
        return self.transform(X)


def detect_outliers(X, method='zscore', threshold=3.0):
    """
    Detect outliers in numerical features using various methods
    
    Args:
        X: Input features (pandas DataFrame or numpy array)
        method: Method to detect outliers ('zscore', 'iqr')
        threshold: Threshold for outlier detection (for zscore and iqr methods)
        
    Returns:
        Boolean mask where True indicates an outlier
    """
    # Convert to pandas DataFrame if numpy array
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)
    
    outlier_mask = np.zeros(len(X), dtype=bool)
    
    if method == 'zscore':
        # Z-score method
        for col in X.select_dtypes(include=np.number).columns:
            z_scores = np.abs(stats.zscore(X[col], nan_policy='omit'))
            outlier_mask = outlier_mask | (z_scores > threshold)
            
    elif method == 'iqr':
        # IQR method
        for col in X.select_dtypes(include=np.number).columns:
            Q1 = X[col].quantile(0.25)
            Q3 = X[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            outlier_mask = outlier_mask | ((X[col] < lower_bound) | (X[col] > upper_bound))
    
    else:
        raise ValueError(f"Unknown outlier detection method: {method}")
    
    return outlier_mask


def select_features(X, y=None, method='importance', n_features=None, threshold=None):
    """
    Select important features for anomaly detection
    
    Args:
        X: Input features (pandas DataFrame or numpy array)
        y: Target variable (for supervised methods)
        method: Feature selection method ('importance', 'mutual_info')
        n_features: Number of features to select
        threshold: Importance threshold for feature selection
        
    Returns:
        Selected features
    """
    # Convert to pandas DataFrame if numpy array
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)
    
    if method == 'importance':
        if y is None:
            # For unsupervised settings, randomly assign some samples as anomalies (5%)
            # This is just to use RandomForest's feature importance
            y = np.zeros(X.shape[0])
            idx = np.random.choice(X.shape[0], size=int(X.shape[0] * 0.05), replace=False)
            y[idx] = 1
        
        # Use RandomForest for feature importance
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        selector = SelectFromModel(model, threshold=threshold, max_features=n_features)
        selector.fit(X, y)
        selected_features = X.columns[selector.get_support()]
        
    elif method == 'mutual_info':
        if y is None:
            raise ValueError("Mutual information method requires a target variable")
        
        # Use mutual information for feature selection
        mi_scores = mutual_info_classif(X, y, random_state=42)
        sorted_idx = np.argsort(mi_scores)[::-1]
        
        if n_features:
            selected_features = X.columns[sorted_idx[:n_features]]
        elif threshold:
            selected_features = X.columns[mi_scores > threshold]
        else:
            # Default to top 10 features
            selected_features = X.columns[sorted_idx[:10]]
            
    else:
        raise ValueError(f"Unknown feature selection method: {method}")
    
    return selected_features


def select_features_by_entropy(X, threshold=0.1):
    """
    Select features based on entropy measures.
    
    Higher entropy features typically contain more information (and potentially more signal
    about anomalies).
    
    Args:
        X: Input features (pandas DataFrame)
        threshold: Minimum entropy threshold for feature selection
        
    Returns:
        List of selected features and dictionary of entropy values
    """
    entropy_values = {}
    selected_features = []
    
    # Calculate entropy for each numerical feature
    for col in X.select_dtypes(include=np.number).columns:
        # Discretize continuous values for entropy calculation
        hist, _ = np.histogram(X[col], bins=20, density=True)
        # Add small epsilon to avoid log(0)
        hist = hist + 1e-10
        hist = hist / np.sum(hist)
        feature_entropy = -np.sum(hist * np.log2(hist))
        entropy_values[col] = feature_entropy
        
        if feature_entropy >= threshold:
            selected_features.append(col)
    
    return selected_features, entropy_values


def compare_distributions(normal_data, anomaly_data, feature):
    """
    Compare distributions of a feature between normal and anomalous data points.
    
    Args:
        normal_data: DataFrame containing normal instances
        anomaly_data: DataFrame containing anomalous instances
        feature: Feature name to compare
        
    Returns:
        Dictionary with distribution statistics and KL divergence
    """
    # Calculate basic statistics
    normal_mean = normal_data[feature].mean()
    normal_std = normal_data[feature].std()
    anomaly_mean = anomaly_data[feature].mean()
    anomaly_std = anomaly_data[feature].std()
    
    # Calculate histograms
    normal_hist, normal_bins = np.histogram(normal_data[feature], bins=20, density=True)
    anomaly_hist, _ = np.histogram(anomaly_data[feature], bins=normal_bins, density=True)
    
    # Add small epsilon to avoid division by zero in KL divergence
    normal_hist = normal_hist + 1e-10
    normal_hist = normal_hist / np.sum(normal_hist)
    anomaly_hist = anomaly_hist + 1e-10
    anomaly_hist = anomaly_hist / np.sum(anomaly_hist)
    
    # Calculate KL divergence
    kl_divergence = stats.entropy(anomaly_hist, normal_hist)
    
    return {
        'normal_mean': normal_mean,
        'normal_std': normal_std,
        'anomaly_mean': anomaly_mean,
        'anomaly_std': anomaly_std,
        'kl_divergence': kl_divergence
    }


def assess_data_balance(y, positive_label=1):
    """
    Assess class balance in the dataset and suggest appropriate thresholds.
    
    Args:
        y: Target labels
        positive_label: The label value that represents anomalies
        
    Returns:
        Dictionary with balance metrics and suggestions
    """
    total = len(y)
    anomalies = np.sum(y == positive_label)
    normal = total - anomalies
    anomaly_ratio = anomalies / total
    
    # Suggested threshold based on class balance
    if anomaly_ratio < 0.01:
        threshold_suggestion = 0.99  # Very imbalanced
    elif anomaly_ratio < 0.05:
        threshold_suggestion = 0.95  # Imbalanced
    elif anomaly_ratio < 0.2:
        threshold_suggestion = 0.9   # Moderately imbalanced
    else:
        threshold_suggestion = 0.8   # Relatively balanced
    
    return {
        'total_samples': total,
        'normal_samples': normal,
        'anomaly_samples': anomalies,
        'anomaly_ratio': anomaly_ratio,
        'suggested_threshold_percentile': threshold_suggestion,
        'suggested_sampling': 'undersampling' if anomaly_ratio < 0.1 else 'none'
    }


def feature_anomaly_correlation(X, y, method='mutual_info'):
    """
    Calculate correlation between features and anomaly status.
    
    Args:
        X: Features
        y: Target (anomaly labels)
        method: Correlation method ('mutual_info', 'point_biserial')
        
    Returns:
        DataFrame with features and their correlation scores
    """
    from scipy.stats import pointbiserialr
    
    correlations = {}
    
    if method == 'mutual_info':
        mi_scores = mutual_info_classif(X, y, random_state=42)
        for i, col in enumerate(X.columns):
            correlations[col] = mi_scores[i]
    
    elif method == 'point_biserial':
        for col in X.columns:
            correlation, p_value = pointbiserialr(X[col], y)
            correlations[col] = correlation
    
    # Convert to DataFrame and sort
    corr_df = pd.DataFrame(correlations.items(), columns=['Feature', 'Correlation'])
    corr_df = corr_df.sort_values('Correlation', ascending=False)
    
    return corr_df


def engineer_kdd_cup_features(df):
    """
    Create domain-specific features for the KDD Cup dataset
    
    Args:
        df: KDD Cup dataset as pandas DataFrame
        
    Returns:
        DataFrame with additional engineered features
    """
    # Create a copy to avoid modifying the original
    df = df.copy()
    
    # Duration-based features
    df['log_duration'] = np.log1p(df['duration'])
    
    # Traffic-based features
    df['bytes_per_second'] = (df['src_bytes'] + df['dst_bytes']) / (df['duration'] + 1)
    df['bytes_ratio'] = df['src_bytes'] / (df['dst_bytes'] + 1)
    
    # Connection-based features
    df['error_rate'] = df['serror_rate'] + df['rerror_rate']
    
    # Count-based features
    df['total_connections'] = df['count'] + df['srv_count']
    df['connection_ratio'] = df['count'] / (df['srv_count'] + 1)
    
    # Service-based features
    df['same_srv_diff_ratio'] = df['same_srv_rate'] / (df['diff_srv_rate'] + 0.01)
    
    # Root access features
    df['root_access_rate'] = df['root_shell'] + df['su_attempted']
    
    return df


def calculate_entropy_matrix(X):
    """
    Calculate entropy for each feature and pairwise mutual information between features.
    
    This provides insights into information content and redundancy in the feature space,
    which is crucial for information-theoretic anomaly detection.
    
    Args:
        X: Input features (pandas DataFrame)
        
    Returns:
        Dictionary with entropy values and mutual information matrix
    """
    # Discretize data for entropy calculation
    n_bins = 20
    X_binned = pd.DataFrame()
    entropy_values = {}
    
    # Discretize each feature and calculate its entropy
    for col in X.select_dtypes(include=np.number).columns:
        # Bin the data
        X_binned[col], bins = pd.cut(X[col], bins=n_bins, retbins=True, labels=False)
        
        # Calculate entropy
        hist = np.histogram(X_binned[col], bins=n_bins, density=True)[0]
        hist = hist + 1e-10  # Add small epsilon to avoid log(0)
        hist = hist / np.sum(hist)
        entropy_values[col] = -np.sum(hist * np.log2(hist))
    
    # Calculate pairwise mutual information
    n_features = len(X_binned.columns)
    mi_matrix = np.zeros((n_features, n_features))
    
    for i, col1 in enumerate(X_binned.columns):
        for j, col2 in enumerate(X_binned.columns):
            if i == j:
                mi_matrix[i, j] = entropy_values[col1]  # Diagonal contains self-entropy
            else:
                # Calculate joint probabilities
                joint_hist = np.histogram2d(
                    X_binned[col1], X_binned[col2], 
                    bins=[n_bins, n_bins], density=True
                )[0]
                
                # Add small epsilon and normalize
                joint_hist = joint_hist + 1e-10
                joint_hist = joint_hist / np.sum(joint_hist)
                
                # Calculate joint entropy
                joint_entropy = -np.sum(joint_hist * np.log2(joint_hist))
                
                # Mutual information = H(X) + H(Y) - H(X,Y)
                mi_matrix[i, j] = entropy_values[col1] + entropy_values[col2] - joint_entropy
    
    return {
        'feature_entropy': entropy_values,
        'mutual_information_matrix': mi_matrix,
        'feature_names': X_binned.columns.tolist()
    } 
