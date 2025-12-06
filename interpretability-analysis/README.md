# Neural Network Interpretability Analysis for Gas Sensing

Feature importance analysis using multiple interpretability methods on semiconductor gas sensor data.

## Overview

This project investigates different methods for analyzing feature importance in neural networks applied to gas concentration prediction. Four methods are compared:

1. **Neural Network Weight Analysis (МАВНС)** - Analyzes connection weights between layers
2. **Deep Taylor Decomposition (МГРТ)** - Relevance propagation method based on Taylor decomposition
3. **Permutation Feature Importance (МПЗВП)** - Measures performance drop when features are shuffled
4. **Fixed-value Feature Importance (МФЗВП)** - Measures performance drop when features are fixed to constant values

## Results

Best performance achieved with 50% feature selection:

| Method | MSE (ppm) | R² |
|--------|-----------|-----|
| Deep Taylor Decomposition | 23.20±0.96 | 0.776±0.010 |
| Weight Analysis | 23.36±0.80 | 0.779±0.009 |
| Permutation | 24.00±2.08 | 0.770±0.020 |
| Fixed Value | 26.72±2.40 | 0.742±0.023 |
| No Selection (Baseline) | 25.92±1.12 | 0.755±0.017 |

**Key Finding:** Methods based on neural network weight analysis (Weight Analysis and Deep Taylor) outperform simple reference methods.

## Project Structure

```
interpretability-analysis/
├── notebooks/                  # Jupyter notebooks
│   ├── deep_taylor_decomposition.ipynb
│   ├── model_training.ipynb
│   └── data_visualization.ipynb
├── src/                       # Source code
│   ├── models.py             # Neural network models
│   ├── deep_taylor.py        # Deep Taylor implementation
│   └── weight_analysis.py    # Weight analysis & other methods
└── results/                   # Results and figures
```

## Usage

### Training a Model

```python
from src.models import GasSensorMLP, create_dataloaders, ModelTrainer
import torch

# Load data
x = torch.load('data/x.pt')
y = torch.load('data/y.pt')

# Create data loaders
train_loader, val_loader, test_loader, train_dataset = create_dataloaders(
    x, y, train_size=0.6, val_size=0.2, batch_size=8
)

# Initialize and train model
model = GasSensorMLP(input_size=550, hidden_size=32, output_size=2)
trainer = ModelTrainer(model, learning_rate=0.001, patience=1000)
history = trainer.train(train_loader, val_loader, num_epochs=5000)

# Evaluate
mse, r2 = trainer.evaluate(test_loader, train_dataset)
print(f"Test MSE: {mse:.2f} ppm, R²: {r2:.3f}")
```

### Computing Feature Importance

```python
from src.deep_taylor import compute_relevance_scores, select_top_features
from src.weight_analysis import weight_analysis_importance

# Deep Taylor Decomposition
dtd_scores = compute_relevance_scores(model, train_loader, normalize=True)

# Weight Analysis
wa_scores = weight_analysis_importance(model, normalize=True)

# Select top 50% features
n_features = int(0.5 * 550)
selected_features_dtd = select_top_features(dtd_scores, n_features=n_features)
selected_features_wa = select_top_features(wa_scores, n_features=n_features)
```

### Visualizing Results

```python
from src.deep_taylor import visualize_relevance
import matplotlib.pyplot as plt

# Visualize top features
visualize_relevance(dtd_scores, top_n=50)
```

## Methods Description

### Deep Taylor Decomposition (МГРТ)

Decomposes the neural network's output by backpropagating relevance scores from output to input, based on first-order Taylor expansion. Provides smooth, interpretable relevance scores.

**Advantages:**
- Considers feature interactions
- Theoretically grounded
- Produces smooth importance curves

### Weight Analysis (МАВНС)

Analyzes the magnitude of connection weights between input and output neurons through hidden layers.

**Formula:**
```
R_ik = Σ_j |w_ij × w_jk|
```

**Advantages:**
- Fast computation
- No need for data samples
- Effective for shallow networks

### Permutation Importance (МПЗВП)

Randomly shuffles each feature and measures the resulting performance decrease.

**Advantages:**
- Model-agnostic
- Easy to interpret
- Captures non-linear relationships

**Disadvantages:**
- Computationally expensive
- High variance

### Fixed-value Importance (МФЗВП)

Fixes each feature to its mean value and measures performance decrease.

**Disadvantages:**
- Doesn't preserve feature distribution
- Can be misleading for correlated features

## Requirements

- Python 3.8+
- PyTorch 1.10+
- NumPy, pandas, scikit-learn
- matplotlib, seaborn

## Citation

```bibtex
@thesis{vitiugova2023interpretability,
  title={Neural Network Weight Analysis for Data Processing in Physics},
  author={Vitiugova, Julia M.},
  year={2023},
  school={Lomonosov Moscow State University},
  type={Bachelor's Thesis}
}
```

## References

1. Montavon, G., Lapuschkin, S., Binder, A., Samek, W., & Müller, K. R. (2017). "Explaining nonlinear classification decisions with deep Taylor decomposition". Pattern Recognition, 65, 211-222.

2. Gevrey, M., Dimopoulos, I., & Lek, S. (2003). "Review and comparison of methods to study the contribution of variables in artificial neural network models". Ecological Modelling, 160(3-4), 249-264.
