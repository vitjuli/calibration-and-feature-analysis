"""
Neural Network Interpretability Analysis Package

This package provides tools for analyzing feature importance in neural networks
applied to gas sensor data.
"""

from .models import GasSensorMLP, SensorDataset, ModelTrainer, create_dataloaders
from .deep_taylor import (
    feature_relevance,
    compute_relevance_scores,
    select_top_features,
    RelevancePropagation,
    visualize_relevance,
)
from .weight_analysis import (
    weight_analysis_importance,
    permutation_importance,
    fixed_value_importance,
    compare_methods,
    select_features_by_threshold,
    visualize_comparison,
)

__version__ = "1.0.0"
__author__ = "Julia Vitiugova"
__email__ = "vityugova.julia@physics.msu.ru"

__all__ = [
    # Models
    "GasSensorMLP",
    "SensorDataset",
    "ModelTrainer",
    "create_dataloaders",
    # Deep Taylor
    "feature_relevance",
    "compute_relevance_scores",
    "select_top_features",
    "RelevancePropagation",
    "visualize_relevance",
    # Weight Analysis
    "weight_analysis_importance",
    "permutation_importance",
    "fixed_value_importance",
    "compare_methods",
    "select_features_by_threshold",
    "visualize_comparison",
]
