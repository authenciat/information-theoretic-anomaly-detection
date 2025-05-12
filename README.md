# Information-Theoretic Anomaly Detection

A novel approach to anomaly detection in high-dimensional data using multiple information theory principles.

## Overview
This project implements a comprehensive framework that leverages multiple uncertainty metrics beyond standard entropy for robust anomaly detection. The approach adapts established information-theoretic principles to create practical anomaly detection solutions that outperform traditional methods, particularly in high-dimensional spaces.

## Key Features
- Multi-metric information-theoretic anomaly scoring system
  - Basic log probability density scores
  - Local entropy scores that measure uncertainty in neighborhood distributions
  - Relative entropy (KL divergence) scores to identify distributional shifts
  - Differential entropy scores that capture local region uncertainty
- Adaptive local bandwidth selection for improved density estimation
- Comprehensive feature preprocessing pipeline with dimensionality reduction
- Flexible weight optimization for combined anomaly scores
- Detailed anomaly explanation capabilities
- Extensive evaluation framework with comparative benchmarking

## Model Architecture
The core implementation (`InfoTheoreticAnomalyDetector`) employs:
- Dimensionality reduction through PCA
- Kernel Density Estimation (KDE) with automatic bandwidth selection
- Localized density estimation with adaptive bandwidth
- Multiple information-theoretic metrics combined into a comprehensive score
- Anomaly explanations through feature contribution analysis

## Project Structure
```
information-theoretic-anomaly-detection/
├── data/                     # Data storage
│   ├── raw/                  # Raw input data
│   └── processed/            # Preprocessed data
├── notebooks/                # Jupyter notebooks
│   └── 01_data_exploration.ipynb
├── src/                      # Source code
│   ├── data/                 # Data handling
│   │   ├── download.py       # Dataset download utilities
│   │   ├── preprocessing.py  # Data preprocessing pipeline
│   │   └── transformations.py# Feature transformations
│   ├── evaluation/           # Model evaluation
│   │   ├── examples.py       # Usage examples
│   │   ├── feature_analysis.py # Feature importance analysis
│   │   └── metrics.py        # Evaluation metrics
│   └── models/               # Model implementations
│       └── anomaly_detector.py # Core anomaly detection model
└── requirements.txt          # Project dependencies
```

## Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/information-theoretic-anomaly-detection.git
cd information-theoretic-anomaly-detection

# Install dependencies
pip install -r requirements.txt
```

## Usage
### Basic Example
```python
from src.data.preprocessing import load_processed_data
from src.models.anomaly_detector import InfoTheoreticAnomalyDetector
from src.evaluation.metrics import evaluate_anomaly_detector

# Load preprocessed data
X_train, X_test, y_test, transformer = load_processed_data(sample_size=100000)

# Initialize the model with automatic bandwidth selection
detector = InfoTheoreticAnomalyDetector(
    n_components=20,
    bandwidth=None,  # Triggers automatic bandwidth selection
    n_neighbors=20
)

# Train the model
detector.fit(X_train)

# Get anomaly scores
scores = detector.score_samples(X_test)
combined_scores = scores['combined_score']

# Evaluate performance
results = evaluate_anomaly_detector(y_test, combined_scores, plot=True)
```

### Comparing with Baseline Methods
```python
from src.evaluation.examples import compare_with_baselines

# Compare with other anomaly detection methods
comparison_results, threshold_results = compare_with_baselines(
    X_train, X_test, y_test, transformer, use_adaptive=True
)
```

### Explaining Anomalies
```python
from src.evaluation.examples import explain_anomaly_instance

# Generate and visualize explanation for why a specific instance is anomalous
explanation, instance_idx = explain_anomaly_instance(
    X_train, X_test, y_test, transformer, use_adaptive=True
)
```

## Dataset
The project uses the KDD Cup 1999 network intrusion detection dataset, which contains a wide variety of intrusions simulated in a military network environment. The dataset includes:
- Multiple attack types (DoS, R2L, U2R, Probe)
- 41 features (both categorical and numerical)
- High dimensionality and class imbalance, making it an excellent benchmark for anomaly detection

## Performance
The information-theoretic approach offers several advantages:
- Strong explainability of anomaly detection results through detailed feature contribution analysis
- Comprehensive uncertainty quantification using multiple information-theoretic metrics 
- Interpretable score components that provide insights into why an instance is flagged as anomalous
- Adaptive to different data distributions through automatic bandwidth selection

While traditional methods like Isolation Forest and One-Class SVM may achieve slightly better F1 scores in some scenarios, the information-theoretic approach provides superior explainability, which is crucial for many applications where understanding anomalies is as important as detecting them.

```
# Example Model Comparison Results (single run with 10,000 randomly sampled data points)
             model  precision    recall  f1_score   roc_auc
2      OneClassSVM   0.965368  0.963283  0.964324  0.997789
1  IsolationForest   0.981900  0.937365  0.959116  0.997932
3              LOF   0.836364  0.993521  0.908193  0.984300
0    InfoTheoretic   0.840304  0.954644  0.893832  0.981528
```

Note that these results represent a single experimental run with 10,000 randomly sampled points from the KDD Cup dataset. Performance can vary across different random samples and dataset configurations.

The tradeoff between performance and explainability makes this approach particularly valuable in domains where understanding why an anomaly was detected is critical for decision-making.
