"""
Neural Network Calibration Methods for Gas Sensor Applications

Implementation of calibration techniques adapted from:
Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017).
"On Calibration of Modern Neural Networks". ICML.

Adapted for regression tasks and sensor drift conditions.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional
from scipy.optimize import minimize


class TemperatureScaling:
    """
    Temperature scaling calibration method.

    Scales the logits by a learned temperature parameter to improve
    calibration of probability estimates.
    """

    def __init__(self):
        """Initialize temperature scaling."""
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def fit(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        lr: float = 0.01,
        max_iter: int = 50
    ) -> float:
        """
        Fit temperature parameter on validation set.

        Args:
            logits: Model outputs (before softmax)
            labels: True labels
            lr: Learning rate
            max_iter: Maximum iterations

        Returns:
            Optimal temperature value
        """
        criterion = nn.MSELoss()
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def closure():
            optimizer.zero_grad()
            loss = criterion(logits / self.temperature, labels)
            loss.backward()
            return loss

        optimizer.step(closure)

        return self.temperature.item()

    def calibrate(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply temperature scaling to logits.

        Args:
            logits: Model outputs

        Returns:
            Calibrated outputs
        """
        return logits / self.temperature


class VectorScaling:
    """
    Vector scaling - extends temperature scaling with per-class scaling.
    """

    def __init__(self, num_outputs: int = 2):
        """
        Initialize vector scaling.

        Args:
            num_outputs: Number of output dimensions
        """
        self.W = nn.Parameter(torch.ones(num_outputs))
        self.b = nn.Parameter(torch.zeros(num_outputs))

    def fit(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        lr: float = 0.01,
        max_iter: int = 50
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit scaling parameters.

        Args:
            logits: Model outputs
            labels: True labels
            lr: Learning rate
            max_iter: Maximum iterations

        Returns:
            Optimal W and b parameters
        """
        criterion = nn.MSELoss()
        optimizer = torch.optim.LBFGS(
            [self.W, self.b],
            lr=lr,
            max_iter=max_iter
        )

        def closure():
            optimizer.zero_grad()
            calibrated = logits * self.W + self.b
            loss = criterion(calibrated, labels)
            loss.backward()
            return loss

        optimizer.step(closure)

        return self.W.detach().numpy(), self.b.detach().numpy()

    def calibrate(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply vector scaling.

        Args:
            logits: Model outputs

        Returns:
            Calibrated outputs
        """
        return logits * self.W + self.b


class MatrixScaling:
    """
    Matrix scaling - full affine transformation of outputs.
    """

    def __init__(self, num_outputs: int = 2):
        """
        Initialize matrix scaling.

        Args:
            num_outputs: Number of output dimensions
        """
        self.W = nn.Parameter(torch.eye(num_outputs))
        self.b = nn.Parameter(torch.zeros(num_outputs))

    def fit(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        lr: float = 0.01,
        max_iter: int = 50
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit matrix scaling parameters.

        Args:
            logits: Model outputs
            labels: True labels
            lr: Learning rate
            max_iter: Maximum iterations

        Returns:
            Optimal W matrix and b vector
        """
        criterion = nn.MSELoss()
        optimizer = torch.optim.LBFGS(
            [self.W, self.b],
            lr=lr,
            max_iter=max_iter
        )

        def closure():
            optimizer.zero_grad()
            calibrated = torch.matmul(logits, self.W.t()) + self.b
            loss = criterion(calibrated, labels)
            loss.backward()
            return loss

        optimizer.step(closure)

        return self.W.detach().numpy(), self.b.detach().numpy()

    def calibrate(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply matrix scaling.

        Args:
            logits: Model outputs

        Returns:
            Calibrated outputs
        """
        return torch.matmul(logits, self.W.t()) + self.b


class IsotonicRegression:
    """
    Isotonic regression calibration for regression problems.
    """

    def __init__(self):
        """Initialize isotonic regression."""
        self.calibrators = []

    def fit(
        self,
        predictions: np.ndarray,
        targets: np.ndarray
    ):
        """
        Fit isotonic regression for each output dimension.

        Args:
            predictions: Model predictions
            targets: True targets
        """
        from sklearn.isotonic import IsotonicRegression as SklearnIR

        n_outputs = predictions.shape[1] if len(predictions.shape) > 1 else 1

        if n_outputs == 1:
            predictions = predictions.reshape(-1, 1)
            targets = targets.reshape(-1, 1)

        self.calibrators = []
        for i in range(n_outputs):
            ir = SklearnIR(out_of_bounds='clip')
            ir.fit(predictions[:, i], targets[:, i])
            self.calibrators.append(ir)

    def calibrate(self, predictions: np.ndarray) -> np.ndarray:
        """
        Apply isotonic regression calibration.

        Args:
            predictions: Model predictions

        Returns:
            Calibrated predictions
        """
        n_outputs = len(self.calibrators)
        calibrated = np.zeros_like(predictions)

        if len(predictions.shape) == 1:
            predictions = predictions.reshape(-1, 1)

        for i in range(n_outputs):
            calibrated[:, i] = self.calibrators[i].predict(predictions[:, i])

        return calibrated


def expected_calibration_error(
    predictions: np.ndarray,
    targets: np.ndarray,
    n_bins: int = 10
) -> float:
    """
    Compute Expected Calibration Error (ECE) for regression.

    Args:
        predictions: Model predictions
        targets: True targets
        n_bins: Number of bins

    Returns:
        ECE value
    """
    # Bin predictions
    bin_boundaries = np.linspace(
        predictions.min(),
        predictions.max(),
        n_bins + 1
    )

    ece = 0.0
    total_samples = len(predictions)

    for i in range(n_bins):
        # Find samples in this bin
        in_bin = (predictions >= bin_boundaries[i]) & \
                 (predictions < bin_boundaries[i + 1])

        if in_bin.sum() == 0:
            continue

        # Compute average prediction and target in bin
        avg_pred = predictions[in_bin].mean()
        avg_target = targets[in_bin].mean()

        # Add weighted difference to ECE
        bin_size = in_bin.sum()
        ece += (bin_size / total_samples) * np.abs(avg_pred - avg_target)

    return ece


def calibration_curve(
    predictions: np.ndarray,
    targets: np.ndarray,
    n_bins: int = 10
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute calibration curve for visualization.

    Args:
        predictions: Model predictions
        targets: True targets
        n_bins: Number of bins

    Returns:
        bin_means: Mean predicted value in each bin
        bin_true_means: Mean true value in each bin
        bin_counts: Number of samples in each bin
    """
    bin_boundaries = np.linspace(
        predictions.min(),
        predictions.max(),
        n_bins + 1
    )

    bin_means = np.zeros(n_bins)
    bin_true_means = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)

    for i in range(n_bins):
        in_bin = (predictions >= bin_boundaries[i]) & \
                 (predictions < bin_boundaries[i + 1])

        if in_bin.sum() > 0:
            bin_means[i] = predictions[in_bin].mean()
            bin_true_means[i] = targets[in_bin].mean()
            bin_counts[i] = in_bin.sum()

    return bin_means, bin_true_means, bin_counts


def visualize_calibration(
    predictions: np.ndarray,
    targets: np.ndarray,
    calibrated_predictions: np.ndarray = None,
    n_bins: int = 10
):
    """
    Visualize calibration before and after calibration.

    Args:
        predictions: Uncalibrated predictions
        targets: True targets
        calibrated_predictions: Calibrated predictions (optional)
        n_bins: Number of bins for calibration curve
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Calibration curve
    bin_means, bin_true_means, _ = calibration_curve(
        predictions.flatten(),
        targets.flatten(),
        n_bins
    )

    axes[0].plot([predictions.min(), predictions.max()],
                 [predictions.min(), predictions.max()],
                 'k--', label='Perfect calibration')
    axes[0].plot(bin_means, bin_true_means, 'o-', label='Before calibration')

    if calibrated_predictions is not None:
        bin_means_cal, bin_true_means_cal, _ = calibration_curve(
            calibrated_predictions.flatten(),
            targets.flatten(),
            n_bins
        )
        axes[0].plot(bin_means_cal, bin_true_means_cal, 's-',
                     label='After calibration')

    axes[0].set_xlabel('Mean Predicted Value')
    axes[0].set_ylabel('Mean True Value')
    axes[0].set_title('Calibration Curve')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Prediction scatter
    axes[1].scatter(predictions, targets, alpha=0.5, s=10,
                   label='Before calibration')

    if calibrated_predictions is not None:
        axes[1].scatter(calibrated_predictions, targets, alpha=0.5, s=10,
                       label='After calibration')

    axes[1].plot([targets.min(), targets.max()],
                 [targets.min(), targets.max()],
                 'k--', label='Perfect prediction')
    axes[1].set_xlabel('Predicted Value')
    axes[1].set_ylabel('True Value')
    axes[1].set_title('Predictions vs Targets')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def compare_calibration_methods(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader
) -> dict:
    """
    Compare different calibration methods.

    Args:
        model: Trained neural network
        val_loader: Validation data for fitting calibration
        test_loader: Test data for evaluation

    Returns:
        results: Dictionary with calibration results
    """
    # Get validation predictions
    model.eval()
    val_preds = []
    val_targets = []

    with torch.no_grad():
        for x, y in val_loader:
            preds = model(x.float())
            val_preds.append(preds)
            val_targets.append(y)

    val_preds = torch.cat(val_preds, dim=0)
    val_targets = torch.cat(val_targets, dim=0)

    # Get test predictions
    test_preds = []
    test_targets = []

    with torch.no_grad():
        for x, y in test_loader:
            preds = model(x.float())
            test_preds.append(preds)
            test_targets.append(y)

    test_preds = torch.cat(test_preds, dim=0)
    test_targets = torch.cat(test_targets, dim=0)

    results = {
        'uncalibrated': {
            'predictions': test_preds.numpy(),
            'ece': expected_calibration_error(
                test_preds.numpy().flatten(),
                test_targets.numpy().flatten()
            )
        }
    }

    # Temperature scaling
    temp_scaler = TemperatureScaling()
    temp_scaler.fit(val_preds, val_targets)
    temp_calibrated = temp_scaler.calibrate(test_preds)

    results['temperature'] = {
        'predictions': temp_calibrated.detach().numpy(),
        'ece': expected_calibration_error(
            temp_calibrated.detach().numpy().flatten(),
            test_targets.numpy().flatten()
        )
    }

    # Vector scaling
    vec_scaler = VectorScaling(num_outputs=val_preds.shape[1])
    vec_scaler.fit(val_preds, val_targets)
    vec_calibrated = vec_scaler.calibrate(test_preds)

    results['vector'] = {
        'predictions': vec_calibrated.detach().numpy(),
        'ece': expected_calibration_error(
            vec_calibrated.detach().numpy().flatten(),
            test_targets.numpy().flatten()
        )
    }

    return results
