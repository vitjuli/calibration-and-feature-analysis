"""
Deep Taylor Decomposition for Neural Network Interpretability

Implementation of Deep Taylor Decomposition method for analyzing
feature importance in neural networks applied to gas sensor data.

Reference:
    Montavon, G., Lapuschkin, S., Binder, A., Samek, W., & Müller, K. R. (2017).
    "Explaining nonlinear classification decisions with deep Taylor decomposition".
    Pattern Recognition, 65, 211-222.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple


def feature_relevance(model: nn.Module, data_loader: torch.utils.data.DataLoader) -> np.ndarray:
    """
    Compute feature relevance using Deep Taylor Decomposition.

    This method decomposes the network's output by backpropagating relevance
    scores from the output layer to the input features.

    Args:
        model: Trained PyTorch model
        data_loader: DataLoader containing input data

    Returns:
        feature_importances: Array of relevance scores for each input feature
    """
    model.eval()
    n_features = 550  # Number of input features
    feature_importances = np.zeros(n_features)

    with torch.enable_grad():
        for data, _ in data_loader:
            x = data.float()
            x = x.requires_grad_(True)

            # Forward pass through the hidden layer
            dense_out = torch.sigmoid(model.dense(x))

            # Compute relevance by averaging over the batch
            dense_out = dense_out.mean(0)
            dense_out.backward(torch.ones_like(dense_out), retain_graph=True)

            # Compute implied importance (gradient * input)
            relevance = (x * x.grad).abs()
            feature_importances += relevance.sum(dim=0).detach().numpy()
            x.grad.zero_()

    return feature_importances


def compute_relevance_scores(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    normalize: bool = True
) -> np.ndarray:
    """
    Compute and optionally normalize relevance scores.

    Args:
        model: Trained PyTorch model
        data_loader: DataLoader containing input data
        normalize: Whether to normalize scores to [0, 1]

    Returns:
        Relevance scores for each feature
    """
    relevance = feature_relevance(model, data_loader)

    if normalize:
        relevance = (relevance - relevance.min()) / (relevance.max() - relevance.min())

    return relevance


def select_top_features(
    relevance_scores: np.ndarray,
    n_features: int = None,
    percentile: float = None
) -> np.ndarray:
    """
    Select top features based on relevance scores.

    Args:
        relevance_scores: Array of feature importance scores
        n_features: Number of top features to select (mutually exclusive with percentile)
        percentile: Percentile threshold for feature selection (0-100)

    Returns:
        indices: Indices of selected features
    """
    if n_features is not None:
        indices = np.argsort(relevance_scores)[-n_features:]
    elif percentile is not None:
        threshold = np.percentile(relevance_scores, percentile)
        indices = np.where(relevance_scores >= threshold)[0]
    else:
        raise ValueError("Either n_features or percentile must be specified")

    return indices


class RelevancePropagation:
    """
    Class for relevance propagation through neural network layers.

    Implements layer-wise relevance propagation for deep Taylor decomposition.
    """

    def __init__(self, model: nn.Module):
        """
        Initialize relevance propagation.

        Args:
            model: Neural network model
        """
        self.model = model
        self.activations = {}
        self.gradients = {}

    def forward_hook(self, layer_name: str):
        """Create forward hook to store activations."""
        def hook(module, input, output):
            self.activations[layer_name] = output.detach()
        return hook

    def backward_hook(self, layer_name: str):
        """Create backward hook to store gradients."""
        def hook(module, grad_input, grad_output):
            self.gradients[layer_name] = grad_output[0].detach()
        return hook

    def register_hooks(self):
        """Register hooks on all layers."""
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                module.register_forward_hook(self.forward_hook(name))
                module.register_backward_hook(self.backward_hook(name))

    def propagate_relevance(
        self,
        input_data: torch.Tensor,
        target_output: int = None
    ) -> torch.Tensor:
        """
        Propagate relevance from output to input.

        Args:
            input_data: Input tensor
            target_output: Index of target output neuron (for classification)

        Returns:
            input_relevance: Relevance scores for input features
        """
        input_data.requires_grad = True
        output = self.model(input_data)

        if target_output is not None:
            output = output[:, target_output]

        # Backpropagate from output
        output.backward(torch.ones_like(output))

        # Get input relevance
        input_relevance = input_data.grad * input_data

        return input_relevance.abs()


def visualize_relevance(
    relevance_scores: np.ndarray,
    feature_names: list = None,
    top_n: int = 20
):
    """
    Visualize top feature relevances.

    Args:
        relevance_scores: Array of relevance scores
        feature_names: List of feature names (optional)
        top_n: Number of top features to display
    """
    import matplotlib.pyplot as plt

    top_indices = np.argsort(relevance_scores)[-top_n:]
    top_scores = relevance_scores[top_indices]

    if feature_names is None:
        feature_names = [f"Feature {i}" for i in top_indices]
    else:
        feature_names = [feature_names[i] for i in top_indices]

    plt.figure(figsize=(10, 6))
    plt.barh(range(top_n), top_scores)
    plt.yticks(range(top_n), feature_names)
    plt.xlabel('Relevance Score')
    plt.title(f'Top {top_n} Feature Importances (Deep Taylor Decomposition)')
    plt.tight_layout()
    plt.show()
