# Neural Network Calibration for Gas Sensor Applications

Calibration techniques for improving probability estimates and model reliability under domain shift and sensor drift conditions.

## Overview

This project adapts neural network calibration techniques (based on Guo et al., 2017) from classification to regression tasks, specifically for gas sensor applications. The methods improve model reliability when facing:

- **Domain shift** - Distribution changes between training and test data
- **Sensor drift** - Temporal changes in sensor characteristics
- **Measurement uncertainty** - Variability in sensor readings

## Calibration Methods

### 1. Temperature Scaling

Scales model outputs by a learned temperature parameter:
```
calibrated_output = output / T
```

**Properties:**
- Single scalar parameter
- Fast to fit
- Effective for well-calibrated models

### 2. Vector Scaling

Per-dimension affine transformation:
```
calibrated_output = W ⊙ output + b
```

**Properties:**
- Separate scaling per output dimension
- More flexible than temperature scaling
- Useful when different outputs have different confidence levels

### 3. Matrix Scaling

Full affine transformation:
```
calibrated_output = W × output + b
```

**Properties:**
- Most flexible
- Captures output correlations
- Risk of overfitting with limited calibration data

### 4. Isotonic Regression

Non-parametric monotonic calibration:
- Fits a piecewise-constant monotonic function
- Flexible but requires more calibration data

## Project Structure

```
calibration/
├── notebooks/                 # Jupyter notebooks
│   └── calibration_experiments.ipynb
├── src/                      # Source code
│   └── calibration_methods.py
└── results/                  # Results and figures
```

## Usage

### Basic Calibration

```python
from src.calibration_methods import (
    TemperatureScaling,
    VectorScaling,
    expected_calibration_error
)
import torch

# Assume we have a trained model and validation/test loaders
model.eval()

# Get validation predictions
val_preds = []
val_targets = []
with torch.no_grad():
    for x, y in val_loader:
        val_preds.append(model(x.float()))
        val_targets.append(y)

val_preds = torch.cat(val_preds)
val_targets = torch.cat(val_targets)

# Fit temperature scaling
temp_scaler = TemperatureScaling()
temp_scaler.fit(val_preds, val_targets)

# Calibrate test predictions
test_preds = []
with torch.no_grad():
    for x, _ in test_loader:
        test_preds.append(model(x.float()))
test_preds = torch.cat(test_preds)

calibrated_preds = temp_scaler.calibrate(test_preds)
```

### Comparing Methods

```python
from src.calibration_methods import compare_calibration_methods

results = compare_calibration_methods(model, val_loader, test_loader)

print("Calibration Results:")
for method, metrics in results.items():
    print(f"{method}: ECE = {metrics['ece']:.4f}")
```

### Visualization

```python
from src.calibration_methods import visualize_calibration

visualize_calibration(
    uncalibrated_preds.numpy(),
    test_targets.numpy(),
    calibrated_preds.numpy(),
    n_bins=10
)
```

## Evaluation Metrics

### Expected Calibration Error (ECE)

Measures the difference between predicted and actual values across bins:

```
ECE = Σ (n_b / n) × |pred_mean_b - target_mean_b|
```

Lower ECE indicates better calibration.

### Calibration Curve

Visual representation of calibration:
- X-axis: Mean predicted value in bin
- Y-axis: Mean actual value in bin
- Perfect calibration: diagonal line

## Sensor Drift Compensation

The calibration methods are particularly effective for handling sensor drift:

1. **Initial Calibration**: Train calibration parameters on fresh sensors
2. **Periodic Recalibration**: Update parameters with recent validation data
3. **Transfer Learning**: Use calibration learned on one sensor for another

## Applications

- **Quality Control**: Ensure consistent predictions across sensor batches
- **Long-term Monitoring**: Compensate for aging sensors
- **Multi-sensor Systems**: Harmonize predictions from different sensor types
- **Safety-critical Systems**: Provide confidence estimates for predictions

## Requirements

- Python 3.8+
- PyTorch 1.10+
- NumPy, scipy, scikit-learn
- matplotlib

## Citation

```bibtex
@article{vitiugova2023calibration,
  title={Neural Network Calibration for Gas Sensor Applications},
  author={Vitiugova, Julia M.},
  journal={In Preparation},
  year={2023}
}
```

## References

1. Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). "On Calibration of Modern Neural Networks". Proceedings of the 34th International Conference on Machine Learning (ICML).

2. Kuleshov, V., Fenner, N., & Ermon, S. (2018). "Accurate Uncertainties for Deep Learning Using Calibrated Regression". Proceedings of the 35th International Conference on Machine Learning (ICML).

3. Naeini, M. P., Cooper, G. F., & Hauskrecht, M. (2015). "Obtaining Well Calibrated Probabilities Using Bayesian Binning". Twenty-Ninth AAAI Conference on Artificial Intelligence.
