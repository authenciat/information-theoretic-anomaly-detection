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