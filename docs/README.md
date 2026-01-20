# Documentation

This folder contains research papers, thesis, and supplementary documentation for the Electronic Nose ML project.

## Contents

### Research Papers

#### 1. Bachelor's Thesis (Russian)
**File:** `thesis_ru.pdf`

**Title:** "Анализ весов нейронной сети при обработке данных в физике" (Neural Network Weight Analysis for Data Processing in Physics)

**Author:** Julia M. Vitiugova

**Institution:** Lomonosov Moscow State University, Faculty of Physics

**Year:** 2023

**Abstract:** This bachelor's thesis investigates methods for analyzing feature importance in neural networks applied to gas sensor data. Four methods are compared: Neural Network Weight Analysis, Deep Taylor Decomposition, Permutation Feature Importance, and Fixed-value Feature Importance. Results show that methods based on weight analysis outperform simpler reference methods, achieving MSE of 23.20 ppm with 50% feature selection.

**Key Chapters:**
- Chapter 2: Physical Problem Statement and Experimental Setup
- Chapter 3: Review of Neural Network Methods
- Chapter 4: Computational Experiments
- Chapter 5: Results and Analysis

#### 2. Calibration Paper (In Preparation)
**File:** `calibration_paper.docx`

**Title:** "Neural Network Calibration for Gas Sensor Applications"

**Status:** Journal article in preparation

**Abstract:** Adaptation of neural network calibration techniques to regression tasks for gas sensing applications. Investigates temperature scaling, vector scaling, and matrix scaling methods for improving model reliability under sensor drift conditions.

### Additional Documentation

#### Method Descriptions

**Deep Taylor Decomposition (DTD)**
- Theoretical foundation based on Taylor expansion
- Layer-wise relevance propagation
- Applications to multilayer perceptrons
- Implementation details and pseudocode

**Weight Analysis Method**
- Connection weight importance calculation
- Mathematical formulation
- Comparison with gradient-based methods
- Limitations and extensions

**Calibration Techniques**
- Temperature scaling for regression
- Evaluation metrics (ECE for regression)
- Sensor drift compensation strategies
- Multi-sensor calibration

#### Experimental Protocols

**Gas Sensor Measurements**
- Sensor fabrication details
- Temperature modulation protocol
- Gas mixing procedure
- Data acquisition parameters

**Neural Network Training**
- Architecture selection rationale
- Hyperparameter tuning
- Early stopping criteria
- Cross-validation strategy

## References

### Key Papers

1. **Montavon, G., Lapuschkin, S., Binder, A., Samek, W., & Müller, K. R. (2017)**
   "Explaining nonlinear classification decisions with deep Taylor decomposition"
   *Pattern Recognition*, 65, 211-222.

2. **Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017)**
   "On Calibration of Modern Neural Networks"
   *Proceedings of the 34th International Conference on Machine Learning (ICML)*

3. **Gevrey, M., Dimopoulos, I., & Lek, S. (2003)**
   "Review and comparison of methods to study the contribution of variables in artificial neural network models"
   *Ecological Modelling*, 160(3-4), 249-264.

### Related Work

**Interpretability Methods:**
- Lundberg & Lee (2017): SHAP values
- Ribeiro et al. (2016): LIME
- Selvaraju et al. (2017): Grad-CAM

**Calibration Methods:**
- Kuleshov et al. (2018): Calibrated regression
- Naeini et al. (2015): Bayesian binning
- Platt (1999): Platt scaling

**Gas Sensor Applications:**
- Krivetskiy et al. (2018): SnO₂ sensor arrays
- Fonollosa et al. (2015): Chemical sensor arrays
- Vergara et al. (2012): Sensor drift dataset

## How to Cite

### BibTeX

```bibtex
@thesis{vitiugova2023interpretability,
  title={Neural Network Weight Analysis for Data Processing in Physics},
  author={Vitiugova, Julia M.},
  year={2023},
  school={Lomonosov Moscow State University},
  type={Bachelor's Thesis},
  address={Moscow, Russia}
}

@article{vitiugova2023calibration,
  title={Neural Network Calibration for Gas Sensor Applications},
  author={Vitiugova, Julia M.},
  journal={In Preparation},
  year={2023}
}
```

### APA Style

Vitiugova, J. M. (2023). *Neural Network Weight Analysis for Data Processing in Physics* [Bachelor's thesis, Lomonosov Moscow State University]. Moscow, Russia.

## Presentations

### Conference Presentations

(Add conference presentations if any)

### Posters

(Add poster presentations if any)

## Contact

For questions about the research or to request additional documentation:

**Julia Vitiugova**
- Email: ivitiugova@gmail.com
- GitHub: [@vitjuli](https://github.com/vitjuli)
- Institution: Lomonosov Moscow State University, Faculty of Physics

**Supervisor:**
Dr. Sergey A. Dolenko
- Laboratory of Mathematical Modeling and Informatics
- Lomonosov Moscow State University

## License

Documentation and research papers are provided for academic and research purposes. Please cite appropriately when using this work.

Code and implementations are licensed under MIT License (see main repository LICENSE file).
