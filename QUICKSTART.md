# Quick Start Guide

Get started with the Electronic Nose ML research projects in 5 minutes!

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/vitjuli/electronic-nose-ml.git
cd electronic-nose-ml
```

### 2. Create Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

Or install in development mode:
```bash
pip install -e .
```

## Quick Examples

### Example 1: Train a Model

```python
import torch
from interpretability_analysis.src import (
    GasSensorMLP,
    create_dataloaders,
    ModelTrainer
)

# Load your data
x = torch.load('data/x.pt')  # Shape: [N, 550]
y = torch.load('data/y.pt')  # Shape: [N, 2]

# Create data loaders
train_loader, val_loader, test_loader, train_dataset = create_dataloaders(
    x, y,
    train_size=0.6,
    val_size=0.2,
    batch_size=8
)

# Initialize model
model = GasSensorMLP(input_size=550, hidden_size=32, output_size=2)

# Train with early stopping
trainer = ModelTrainer(model, learning_rate=0.001, patience=1000)
history = trainer.train(train_loader, val_loader, num_epochs=5000, verbose=True)

# Evaluate
mse, r2 = trainer.evaluate(test_loader, train_dataset)
print(f"Test MSE: {mse:.2f} ppm")
print(f"Test R²: {r2:.3f}")
```

### Example 2: Analyze Feature Importance

```python
from interpretability_analysis.src import (
    compute_relevance_scores,
    weight_analysis_importance,
    select_top_features
)

# Method 1: Deep Taylor Decomposition
dtd_scores = compute_relevance_scores(model, train_loader, normalize=True)

# Method 2: Weight Analysis
wa_scores = weight_analysis_importance(model, normalize=True)

# Select top 50% features
n_features = int(0.5 * 550)
top_features_dtd = select_top_features(dtd_scores, n_features=n_features)
top_features_wa = select_top_features(wa_scores, n_features=n_features)

print(f"Selected {len(top_features_dtd)} features with DTD")
print(f"Selected {len(top_features_wa)} features with Weight Analysis")
```

### Example 3: Visualize Results

```python
import matplotlib.pyplot as plt
from interpretability_analysis.src import visualize_relevance

# Visualize top 50 features
visualize_relevance(dtd_scores, top_n=50)

# Plot importance scores
plt.figure(figsize=(12, 5))
plt.plot(dtd_scores, label='Deep Taylor', alpha=0.7)
plt.plot(wa_scores, label='Weight Analysis', alpha=0.7)
plt.xlabel('Feature Index')
plt.ylabel('Importance Score')
plt.legend()
plt.title('Feature Importance Comparison')
plt.show()
```

### Example 4: Calibrate Model

```python
from calibration.src import (
    TemperatureScaling,
    compare_calibration_methods,
    visualize_calibration
)

# Get model predictions
with torch.no_grad():
    val_preds = []
    val_targets = []
    for x, y in val_loader:
        val_preds.append(model(x.float()))
        val_targets.append(y)

val_preds = torch.cat(val_preds)
val_targets = torch.cat(val_targets)

# Fit temperature scaling
temp_scaler = TemperatureScaling()
temp_scaler.fit(val_preds, val_targets)
print(f"Optimal temperature: {temp_scaler.temperature.item():.3f}")

# Apply to test set
with torch.no_grad():
    test_preds = []
    test_targets = []
    for x, y in test_loader:
        test_preds.append(model(x.float()))
        test_targets.append(y)

test_preds = torch.cat(test_preds)
test_targets = torch.cat(test_targets)

calibrated_preds = temp_scaler.calibrate(test_preds)

# Visualize calibration improvement
visualize_calibration(
    test_preds.numpy(),
    test_targets.numpy(),
    calibrated_preds.detach().numpy()
)
```

## Jupyter Notebooks

Explore the provided notebooks for detailed examples:

### Interpretability Analysis
```bash
cd interpretability-analysis/notebooks
jupyter notebook
```

Open:
- `deep_taylor_decomposition.ipynb` - DTD implementation and experiments
- `model_training.ipynb` - Model training and evaluation
- `data_visualization.ipynb` - Data exploration and visualization

### Calibration
```bash
cd calibration/notebooks
jupyter notebook
```

Open:
- `calibration_experiments.ipynb` - Calibration methods comparison

## Common Tasks

### Load Pretrained Model
```python
model = GasSensorMLP()
model.load_state_dict(torch.load('models/model_weights.pth'))
model.eval()
```

### Save Model
```python
torch.save(model.state_dict(), 'models/my_model.pth')
```

### Compare Multiple Methods
```python
from interpretability_analysis.src import compare_methods

results_df = compare_methods(
    model,
    train_loader,
    methods=['weight', 'permutation', 'fixed']
)
print(results_df.head())
```

### Evaluate Calibration
```python
from calibration.src import expected_calibration_error

ece_before = expected_calibration_error(
    test_preds.numpy().flatten(),
    test_targets.numpy().flatten()
)
ece_after = expected_calibration_error(
    calibrated_preds.detach().numpy().flatten(),
    test_targets.numpy().flatten()
)

print(f"ECE before: {ece_before:.4f}")
print(f"ECE after: {ece_after:.4f}")
print(f"Improvement: {(ece_before - ece_after)/ece_before*100:.1f}%")
```

## Project Structure Overview

```
electronic-nose-ml/
├── interpretability-analysis/    # Feature importance project
│   ├── notebooks/               # Jupyter notebooks
│   ├── src/                    # Python modules
│   │   ├── models.py          # Neural network models
│   │   ├── deep_taylor.py     # Deep Taylor Decomposition
│   │   └── weight_analysis.py # Weight-based methods
│   └── results/               # Experiment results
│
├── calibration/                 # Calibration project
│   ├── notebooks/              # Jupyter notebooks
│   ├── src/                   # Python modules
│   │   └── calibration_methods.py
│   └── results/               # Experiment results
│
├── data/                       # Dataset (not included)
├── models/                     # Saved model weights
├── docs/                       # Documentation and papers
└── requirements.txt            # Dependencies
```

## Next Steps

1. **Read the full documentation:**
   - [Main README](README.md)
   - [Interpretability Analysis README](interpretability-analysis/README.md)
   - [Calibration README](calibration/README.md)
   - [Data README](data/README.md)

2. **Explore the notebooks:**
   - Start with data visualization
   - Try model training
   - Experiment with interpretability methods
   - Test calibration techniques

3. **Run experiments:**
   - Compare different methods
   - Try different hyperparameters
   - Visualize results

4. **Contribute:**
   - See [CONTRIBUTING.md](CONTRIBUTING.md)
   - Report issues on GitHub
   - Share your results

## Troubleshooting

### Import Errors
```bash
# Make sure you're in the right directory
cd electronic-nose-ml

# Install package in development mode
pip install -e .
```

### CUDA/GPU Issues
```python
# Check if CUDA is available
import torch
print(f"CUDA available: {torch.cuda.is_available()}")

# Force CPU if needed
device = torch.device('cpu')
model = model.to(device)
```

### Data Loading Issues
```python
# Check data shapes
print(f"X shape: {x.shape}")  # Should be [N, 550]
print(f"Y shape: {y.shape}")  # Should be [N, 2]

# Check data types
print(f"X dtype: {x.dtype}")  # Should be torch.float32
print(f"Y dtype: {y.dtype}")  # Should be torch.float32
```

## Getting Help

- **Documentation:** Check the README files in each folder
- **Issues:** Open an issue on GitHub
- **Email:** vityugova.julia@physics.msu.ru

## Citation

If you use this code, please cite:

```bibtex
@thesis{vitiugova2023interpretability,
  title={Neural Network Weight Analysis for Data Processing in Physics},
  author={Vitiugova, Julia M.},
  year={2023},
  school={Lomonosov Moscow State University}
}
```

Happy researching! 🚀
