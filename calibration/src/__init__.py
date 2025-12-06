"""
Neural Network Calibration Package

This package provides calibration methods for improving neural network
reliability in gas sensor applications.
"""

from .calibration_methods import (
    TemperatureScaling,
    VectorScaling,
    MatrixScaling,
    IsotonicRegression,
    expected_calibration_error,
    calibration_curve,
    visualize_calibration,
    compare_calibration_methods,
)

__version__ = "1.0.0"
__author__ = "Julia Vitiugova"
__email__ = "vityugova.julia@physics.msu.ru"

__all__ = [
    "TemperatureScaling",
    "VectorScaling",
    "MatrixScaling",
    "IsotonicRegression",
    "expected_calibration_error",
    "calibration_curve",
    "visualize_calibration",
    "compare_calibration_methods",
]
