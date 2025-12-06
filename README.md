# Electronic Nose Machine Learning Research

Research projects on neural network interpretability and calibration for gas sensing applications using semiconductor sensors.

**Author:** Iuliia Vitiugova \
**Institution:** Lomonosov Moscow State University, Faculty of Physics


## Projects

### 1. Interpretability Analysis in Physics Data

Investigation of neural network feature importance methods for gas concentration detection (hydrogen and propane) using semiconductor sensors.

**Key Features:**
- Neural Network Weight Analysis (МАВНС)
- Deep Taylor Decomposition (МГРТ)
- Permutation-based Feature Importance (МПЗВП)
- Fixed-value Feature Importance (МФЗВП)
- Comparison of interpretability methods on real sensor data

**Publication:** *"Comparison of Input Feature Selection Methods Based on Neural Network Weight Analysis"* (2023)

### 2. Neural Network Calibration for Sensor Data

Adaptation of neural network calibration techniques (based on Guo et al., 2017) to gas sensing applications, improving probability estimates and model reliability under domain shift and sensor drift conditions.

**Key Features:**
- Temperature scaling
- Vector scaling
- Matrix scaling
- Calibration under distribution shift
- Sensor drift compensation

## Repository Structure

```
.
├── interpretability-analysis/    # Feature importance analysis project
│   ├── notebooks/               # Jupyter notebooks
│   ├── src/                    # Source code
│   └── results/                # Results and visualizations
├── calibration/                # Neural network calibration project
│   ├── notebooks/              # Jupyter notebooks
│   ├── src/                    # Source code
│   └── results/                # Results and visualizations
├── data/                       # Dataset information (data not included)
├── models/                     # Trained model weights
├── docs/                       # Documentation and publications
└── requirements.txt            # Python dependencies
```

## Experimental Setup

### Data
- **Source:** SnO₂-based semiconductor gas sensors
- **Gases:** Hydrogen (H₂) and Propane (C₃H₈)
- **Measurement period:** March-April 2019
- **Location:** Moscow, Russia
- **Temperature modulation:** 150°C - 500°C
- **Sensors:** 4 independent SnO₂ sensors
- **Concentrations:** 30, 50, 100, 200, 350, 400, 450, 470, 550, 800, 950, 1600 ppm

### Neural Network Architecture
- **Type:** Multilayer Perceptron (MLP)
- **Input layer:** 550 features (sensor resistance time series)
- **Hidden layer:** 32 neurons (sigmoid activation)
- **Output layer:** 2 neurons (gas concentrations)
- **Optimizer:** Adam (lr=0.001)
- **Loss:** Mean Squared Error (MSE)

## Results

### Interpretability Analysis
Best results achieved with 50% feature selection:
- **Deep Taylor Decomposition:** MSE = 23.20±0.96 ppm
- **Weight Analysis Method:** R² = 0.779±0.009

Methods based on neural network weight analysis outperformed reference methods (permutation and fixed-value approaches).

### Calibration
Demonstrated improved probability calibration under sensor drift conditions compared to uncalibrated models.

## Getting Started

### Prerequisites
```bash
Python 3.8+
PyTorch 1.10+
pandas
numpy
matplotlib
scikit-learn
```

### Installation
```bash
git clone https://github.com/vitjuli/electronic-nose-ml.git
cd electronic-nose-ml
pip install -r requirements.txt
```

### Running the Code

#### Interpretability Analysis
```bash
cd interpretability-analysis/notebooks
jupyter notebook feature_importance_analysis.ipynb
```

#### Calibration
```bash
cd calibration/notebooks
jupyter notebook calibration_experiments.ipynb
```
```

## Publications

1. **Vitiugova J.M.** "Comparison of Input Feature Selection Methods Based on Neural Network Weight Analysis" - Bachelor's Thesis, Lomonosov Moscow State University, 2023

2. **Vitiugova J.M.** "Neural Network Calibration for Gas Sensor Applications" - Journal Article (in preparation), 2023

## Related Work

- Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). "On Calibration of Modern Neural Networks". ICML.
- Montavon, G., Lapuschkin, S., Binder, A., Samek, W., & Müller, K. R. (2017). "Explaining nonlinear classification decisions with deep Taylor decomposition". Pattern Recognition, 65, 211-222.


---

**Note:** Raw experimental data is not included in this repository due to size limitations. Please contact the author for data access requests.
