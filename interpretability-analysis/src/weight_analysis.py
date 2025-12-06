"""
Neural Network Weight Analysis for Feature Importance

Implementation of weight-based feature importance methods including:
- Weight Analysis Method (МАВНС)
- Permutation Importance (МПЗВП)
- Fixed-value Importance (МФЗВП)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Callable, List
import pandas as pd


def weight_analysis_importance(
    model: nn.Module,
    normalize: bool = True
) -> np.ndarray:
    """
    Compute feature importance using weight analysis method (МАВНС).

    This method analyzes the connection weights between input and hidden layers
    to determine feature importance.

    Formula:
        R_ik = sum_j(|w_ij * w_jk|) / sum_i sum_j(|w_ij * w_jk|)

    Args:
        model: Trained neural network model
        normalize: Whether to normalize importance scores

    Returns:
        importance_scores: Array of importance values for each input feature
    """
    # Get weights from first layer (input to hidden)
    w_input_hidden = model.dense.weight.data.cpu().numpy()  # Shape: (hidden_size, input_size)

    # Get weights from second layer (hidden to output)
    w_hidden_output = model.out.weight.data.cpu().numpy()  # Shape: (output_size, hidden_size)

    n_inputs = w_input_hidden.shape[1]
    n_hidden = w_input_hidden.shape[0]
    n_outputs = w_hidden_output.shape[0]

    # Calculate importance for each input feature
    importance_scores = np.zeros(n_inputs)

    for i in range(n_inputs):
        for j in range(n_hidden):
            for k in range(n_outputs):
                importance_scores[i] += np.abs(w_input_hidden[j, i] * w_hidden_output[k, j])

    if normalize:
        importance_scores = importance_scores / importance_scores.sum()

    return importance_scores


def permutation_importance(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    criterion: Callable = nn.MSELoss(),
    n_repeats: int = 1
) -> np.ndarray:
    """
    Compute feature importance using permutation method (МПЗВП).

    This method randomly shuffles each feature and measures the decrease
    in model performance.

    Args:
        model: Trained model
        data_loader: DataLoader with validation data
        criterion: Loss function to evaluate model performance
        n_repeats: Number of times to permute each feature

    Returns:
        importance_scores: Array of importance values
    """
    model.eval()

    # Get baseline loss
    baseline_loss = 0.0
    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            y_pred = model(x_batch.float())
            loss = criterion(y_pred, y_batch.float())
            baseline_loss += loss.item()
    baseline_loss /= len(data_loader)

    n_features = next(iter(data_loader))[0].shape[1]
    importance_scores = np.zeros(n_features)

    # Permute each feature
    for feature_idx in range(n_features):
        feature_losses = []

        for _ in range(n_repeats):
            permuted_loss = 0.0

            with torch.no_grad():
                for x_batch, y_batch in data_loader:
                    x_permuted = x_batch.clone()
                    # Shuffle the feature across the batch
                    perm_idx = torch.randperm(x_permuted.shape[0])
                    x_permuted[:, feature_idx] = x_permuted[perm_idx, feature_idx]

                    y_pred = model(x_permuted.float())
                    loss = criterion(y_pred, y_batch.float())
                    permuted_loss += loss.item()

            permuted_loss /= len(data_loader)
            feature_losses.append(permuted_loss)

        # Importance = average increase in loss
        importance_scores[feature_idx] = np.mean(feature_losses) - baseline_loss

    return importance_scores


def fixed_value_importance(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    criterion: Callable = nn.MSELoss(),
    fixed_value: str = 'mean'
) -> np.ndarray:
    """
    Compute feature importance using fixed-value method (МФЗВП).

    This method fixes each feature to a constant value (mean or zero)
    and measures the decrease in model performance.

    Args:
        model: Trained model
        data_loader: DataLoader with validation data
        criterion: Loss function
        fixed_value: 'mean', 'zero', or 'median'

    Returns:
        importance_scores: Array of importance values
    """
    model.eval()

    # Get baseline loss
    baseline_loss = 0.0
    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            y_pred = model(x_batch.float())
            loss = criterion(y_pred, y_batch.float())
            baseline_loss += loss.item()
    baseline_loss /= len(data_loader)

    # Compute feature statistics
    all_data = []
    for x_batch, _ in data_loader:
        all_data.append(x_batch)
    all_data = torch.cat(all_data, dim=0)

    if fixed_value == 'mean':
        feature_values = all_data.mean(dim=0)
    elif fixed_value == 'median':
        feature_values = all_data.median(dim=0)[0]
    elif fixed_value == 'zero':
        feature_values = torch.zeros(all_data.shape[1])
    else:
        raise ValueError(f"Unknown fixed_value: {fixed_value}")

    n_features = all_data.shape[1]
    importance_scores = np.zeros(n_features)

    # Fix each feature to constant value
    for feature_idx in range(n_features):
        fixed_loss = 0.0

        with torch.no_grad():
            for x_batch, y_batch in data_loader:
                x_fixed = x_batch.clone()
                x_fixed[:, feature_idx] = feature_values[feature_idx]

                y_pred = model(x_fixed.float())
                loss = criterion(y_pred, y_batch.float())
                fixed_loss += loss.item()

        fixed_loss /= len(data_loader)
        importance_scores[feature_idx] = fixed_loss - baseline_loss

    return importance_scores


def compare_methods(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    methods: List[str] = None
) -> pd.DataFrame:
    """
    Compare different feature importance methods.

    Args:
        model: Trained model
        data_loader: DataLoader with data
        methods: List of methods to compare ['weight', 'permutation', 'fixed', 'deep_taylor']

    Returns:
        df: DataFrame with importance scores from each method
    """
    if methods is None:
        methods = ['weight', 'permutation', 'fixed']

    results = {}

    if 'weight' in methods:
        results['Weight Analysis'] = weight_analysis_importance(model)

    if 'permutation' in methods:
        results['Permutation'] = permutation_importance(model, data_loader)

    if 'fixed' in methods:
        results['Fixed Value'] = fixed_value_importance(model, data_loader)

    # Create DataFrame
    df = pd.DataFrame(results)
    df.index.name = 'Feature'

    return df


def select_features_by_threshold(
    importance_scores: np.ndarray,
    threshold_method: str = 'mean',
    n_std: float = 1.0
) -> np.ndarray:
    """
    Select features based on threshold.

    Args:
        importance_scores: Array of importance scores
        threshold_method: 'mean', 'median', or 'percentile'
        n_std: Number of standard deviations above mean (for threshold_method='mean')

    Returns:
        selected_indices: Indices of selected features
    """
    if threshold_method == 'mean':
        threshold = importance_scores.mean() + n_std * importance_scores.std()
    elif threshold_method == 'median':
        threshold = np.median(importance_scores)
    else:
        raise ValueError(f"Unknown threshold_method: {threshold_method}")

    selected_indices = np.where(importance_scores >= threshold)[0]

    return selected_indices


def visualize_comparison(importance_df: pd.DataFrame, top_n: int = 50):
    """
    Visualize comparison of different importance methods.

    Args:
        importance_df: DataFrame with importance scores from different methods
        top_n: Number of top features to display
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, axes = plt.subplots(1, len(importance_df.columns), figsize=(15, 5))

    if len(importance_df.columns) == 1:
        axes = [axes]

    for idx, method in enumerate(importance_df.columns):
        scores = importance_df[method].values
        top_indices = np.argsort(scores)[-top_n:]

        axes[idx].plot(top_indices, scores[top_indices], 'o-')
        axes[idx].set_xlabel('Feature Index')
        axes[idx].set_ylabel('Importance Score')
        axes[idx].set_title(method)
        axes[idx].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
