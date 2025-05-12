# Data loading and preprocessing functions 
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from .transformations import AnomalyFeatureTransformer, engineer_kdd_cup_features

# Define column names for the KDD Cup dataset
KDD_COLUMNS = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
    'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
    'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate', 'label'
]

# Define categorical and numeric features
CATEGORICAL_FEATURES = ['protocol_type', 'service', 'flag']
NUMERIC_FEATURES = [col for col in KDD_COLUMNS if col not in CATEGORICAL_FEATURES + ['label']]

# Define attack types mapping
ATTACK_MAPPING = {
    'normal': 'normal',
    'back': 'dos', 'land': 'dos', 'neptune': 'dos', 'pod': 'dos', 
    'smurf': 'dos', 'teardrop': 'dos',
    'buffer_overflow': 'u2r', 'loadmodule': 'u2r', 'perl': 'u2r', 'rootkit': 'u2r',
    'ftp_write': 'r2l', 'guess_passwd': 'r2l', 'imap': 'r2l', 'multihop': 'r2l', 
    'phf': 'r2l', 'spy': 'r2l', 'warezclient': 'r2l', 'warezmaster': 'r2l',
    'ipsweep': 'probe', 'nmap': 'probe', 'portsweep': 'probe', 'satan': 'probe'
}

def load_kdd_data(file_path="data/raw/kddcup.data", sample_size=None, random_state=42):
    """
    Load the KDD Cup dataset and optionally sample it for faster processing.
    
    Args:
        file_path: Path to the KDD Cup data file
        sample_size: If specified, sample this many instances randomly
        random_state: Random seed for reproducibility
        
    Returns:
        DataFrame containing the data
    """
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file {file_path} not found. Run download.py first.")
    
    # Load the data
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path, names=KDD_COLUMNS, header=None)
    
    # Sample if requested
    if sample_size and sample_size < len(df):
        print(f"Sampling {sample_size} instances...")
        df = df.sample(sample_size, random_state=random_state)
    
    print(f"Loaded data with shape: {df.shape}")
    return df

def preprocess_labels(df):
    """
    Preprocess the labels in the KDD Cup dataset.
    
    Args:
        df: DataFrame containing the data
        
    Returns:
        DataFrame with processed labels
    """
    # Extract attack type from labels (remove the '.' if present)
    df['attack_type'] = df['label'].str.replace('.', '', regex=False)
    
    # Map attack types to their categories
    df['attack_category'] = df['attack_type'].map(ATTACK_MAPPING)
    
    # Create binary label (1 for anomaly, 0 for normal)
    df['is_anomaly'] = (df['attack_category'] != 'normal').astype(int)
    
    return df

def prepare_data(df, categorical_features=CATEGORICAL_FEATURES, numeric_features=NUMERIC_FEATURES, 
                engineer_features=True, test_size=0.3, random_state=42):
    """
    Prepare the data for anomaly detection by preprocessing and splitting.
    
    Args:
        df: DataFrame containing the KDD Cup data
        categorical_features: List of categorical features
        numeric_features: List of numeric features
        engineer_features: Whether to create additional engineered features
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
        
    Returns:
        X_train, X_test, y_train, y_test, transformer
    """
    # Preprocess labels
    df = preprocess_labels(df)
    
    # Feature engineering if requested
    if engineer_features:
        print("Engineering additional features...")
        df = engineer_kdd_cup_features(df)
    
    # Create feature and target arrays
    X = df[categorical_features + numeric_features]
    y = df['is_anomaly']
    
    # Split into training and testing sets
    print(f"Splitting data with test_size={test_size}...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Initialize and fit the transformer
    print("Fitting feature transformer...")
    transformer = AnomalyFeatureTransformer(
        categorical_features=categorical_features,
        numeric_features=numeric_features,
        scaling_method='standard',
        categorical_encoding='onehot',
        dimensionality_reduction='pca',
        n_components=20,
        random_state=random_state
    )
    
    # Fit the transformer on the training data only
    X_train_transformed = transformer.fit_transform(X_train)
    
    # Transform the test data
    X_test_transformed = transformer.transform(X_test)
    
    print(f"Transformed data shapes - X_train: {X_train_transformed.shape}, X_test: {X_test_transformed.shape}")
    
    return X_train_transformed, X_test_transformed, y_train, y_test, transformer

def prepare_unsupervised_data(df, categorical_features=CATEGORICAL_FEATURES, 
                             numeric_features=NUMERIC_FEATURES, engineer_features=True, 
                             contamination=0.1, random_state=42):
    """
    Prepare data for unsupervised anomaly detection where only normal samples are used for training.
    
    Args:
        df: DataFrame containing the KDD Cup data
        categorical_features: List of categorical features
        numeric_features: List of numeric features
        engineer_features: Whether to create additional engineered features
        contamination: Expected proportion of anomalies in test data
        random_state: Random seed for reproducibility
        
    Returns:
        X_train_normal, X_test, y_test, transformer
    """
    # Preprocess labels
    df = preprocess_labels(df)
    
    # Feature engineering if requested
    if engineer_features:
        print("Engineering additional features...")
        df = engineer_kdd_cup_features(df)
    
    # Create feature and target arrays
    X = df[categorical_features + numeric_features]
    y = df['is_anomaly']
    
    # Get normal data for training
    normal_data = df[df['is_anomaly'] == 0]
    X_normal = normal_data[categorical_features + numeric_features]
    
    # Split normal data into train and test
    X_train_normal, X_test_normal = train_test_split(
        X_normal, test_size=0.3, random_state=random_state
    )
    
    # Get anomaly data for testing
    anomaly_data = df[df['is_anomaly'] == 1]
    X_anomaly = anomaly_data[categorical_features + numeric_features]
    
    # Sample anomalies to achieve desired contamination
    n_normal_test = len(X_test_normal)
    n_anomaly_test = int(n_normal_test * contamination / (1 - contamination))
    X_anomaly_test = X_anomaly.sample(n=min(n_anomaly_test, len(X_anomaly)), random_state=random_state)
    
    # Combine normal and anomaly test data
    X_test = pd.concat([X_test_normal, X_anomaly_test])
    y_test = pd.Series([0] * len(X_test_normal) + [1] * len(X_anomaly_test))
    
    # Shuffle test data
    test_indices = np.random.permutation(len(X_test))
    X_test = X_test.iloc[test_indices].reset_index(drop=True)
    y_test = y_test.iloc[test_indices].reset_index(drop=True)
    
    # Initialize and fit the transformer
    print("Fitting feature transformer...")
    transformer = AnomalyFeatureTransformer(
        categorical_features=categorical_features,
        numeric_features=numeric_features,
        scaling_method='standard',
        categorical_encoding='onehot',
        dimensionality_reduction='pca',
        n_components=20,
        random_state=random_state
    )
    
    # Fit the transformer on the normal training data only
    X_train_transformed = transformer.fit_transform(X_train_normal)
    
    # Transform the test data
    X_test_transformed = transformer.transform(X_test)
    
    print(f"Transformed data shapes - X_train: {X_train_transformed.shape}, X_test: {X_test_transformed.shape}")
    
    return X_train_transformed, X_test_transformed, y_test, transformer

def load_processed_data(load_unsupervised=False, sample_size=100000, random_state=42):
    """
    Convenience function to load and preprocess the data in one step.
    
    Args:
        load_unsupervised: Whether to prepare data for unsupervised learning
        sample_size: Number of instances to sample
        random_state: Random seed for reproducibility
        
    Returns:
        Prepared data (either supervised or unsupervised)
    """
    # Load the data
    df = load_kdd_data(sample_size=sample_size, random_state=random_state)
    
    # Prepare the data
    if load_unsupervised:
        return prepare_unsupervised_data(df, random_state=random_state)
    else:
        return prepare_data(df, random_state=random_state) 