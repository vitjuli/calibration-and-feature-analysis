"""
Neural Network Models for Gas Sensor Data

Contains model architectures and training utilities for gas concentration
prediction from semiconductor sensor resistance measurements.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, Optional


class SensorDataset(Dataset):
    """
    Dataset class for gas sensor data with normalization.

    Attributes:
        features: Input features (sensor resistance time series)
        labels: Target labels (gas concentrations)
        x_min, x_max: Min/max values for feature normalization
        y_min, y_max: Min/max values for label normalization
    """

    def __init__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        x_min: Optional[torch.Tensor] = None,
        y_min: Optional[torch.Tensor] = None,
        x_max: Optional[torch.Tensor] = None,
        y_max: Optional[torch.Tensor] = None
    ):
        """
        Initialize dataset with optional normalization parameters.

        Args:
            x: Input features
            y: Target labels
            x_min, x_max: Normalization bounds for features (computed if None)
            y_min, y_max: Normalization bounds for labels (computed if None)
        """
        if x_min is None:
            self.y_min = torch.min(y, dim=0)[0]
            self.x_min = torch.min(x, dim=0)[0]
            self.y_max = torch.max(y, dim=0)[0]
            self.x_max = torch.max(x, dim=0)[0]
        else:
            self.y_min = y_min
            self.x_min = x_min
            self.y_max = y_max
            self.x_max = x_max

        self.features = self.normalize(x, self.x_min, self.x_max)
        self.labels = self.normalize(y, self.y_min, self.y_max)

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get item by index."""
        return self.features[idx], self.labels[idx]

    @staticmethod
    def normalize(
        x: torch.Tensor,
        min_vec: torch.Tensor,
        max_vec: torch.Tensor
    ) -> torch.Tensor:
        """
        Normalize data to [0, 1] range.

        Args:
            x: Input tensor
            min_vec: Minimum values
            max_vec: Maximum values

        Returns:
            Normalized tensor
        """
        return (x - min_vec) / (max_vec - min_vec)

    @staticmethod
    def denormalize(
        x: torch.Tensor,
        min_vec: torch.Tensor,
        max_vec: torch.Tensor
    ) -> torch.Tensor:
        """
        Denormalize data back to original scale.

        Args:
            x: Normalized tensor
            min_vec: Minimum values
            max_vec: Maximum values

        Returns:
            Denormalized tensor
        """
        return x * (max_vec - min_vec) + min_vec


class GasSensorMLP(nn.Module):
    """
    Multi-layer Perceptron for gas concentration prediction.

    Architecture:
        - Input layer: 550 features (sensor resistance time series)
        - Hidden layer: 32 neurons with sigmoid activation
        - Output layer: 2 neurons (H2 and C3H8 concentrations)
    """

    def __init__(
        self,
        input_size: int = 550,
        hidden_size: int = 32,
        output_size: int = 2
    ):
        """
        Initialize MLP model.

        Args:
            input_size: Number of input features
            hidden_size: Number of hidden neurons
            output_size: Number of outputs
        """
        super(GasSensorMLP, self).__init__()
        self.dense = nn.Linear(input_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, input_size)

        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        x = torch.sigmoid(self.dense(x))
        return self.out(x)


class ModelTrainer:
    """
    Trainer class for neural network models with early stopping.
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module = nn.MSELoss(),
        optimizer: torch.optim.Optimizer = None,
        learning_rate: float = 0.001,
        patience: int = 1000
    ):
        """
        Initialize trainer.

        Args:
            model: Neural network model
            criterion: Loss function
            optimizer: Optimizer (Adam if None)
            learning_rate: Learning rate
            patience: Early stopping patience (epochs)
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer or torch.optim.Adam(
            model.parameters(),
            lr=learning_rate
        )
        self.patience = patience
        self.best_val_loss = float('inf')
        self.counter = 0
        self.training_history = {
            'train_loss': [],
            'val_loss': []
        }

    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader

        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0

        for x_batch, y_batch in train_loader:
            y_pred = self.model(x_batch.float())
            loss = self.criterion(y_pred.float(), y_batch.float())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    def validate(self, val_loader: DataLoader) -> float:
        """
        Validate model.

        Args:
            val_loader: Validation data loader

        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                y_pred = self.model(x_batch.float())
                loss = self.criterion(y_pred, y_batch)
                total_loss += loss.item()

        return total_loss / len(val_loader)

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 5000,
        verbose: bool = True
    ) -> dict:
        """
        Train model with early stopping.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Maximum number of epochs
            verbose: Whether to print progress

        Returns:
            Training history dictionary
        """
        for epoch in range(num_epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)

            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)

            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.counter = 0
                # Save best model
                self.best_model_state = self.model.state_dict().copy()
            else:
                self.counter += 1

            if self.counter >= self.patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch + 1}")
                # Restore best model
                self.model.load_state_dict(self.best_model_state)
                break

            if verbose and (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}/{num_epochs}, "
                      f"Train Loss: {train_loss:.6f}, "
                      f"Val Loss: {val_loss:.6f}")

        return self.training_history

    def evaluate(
        self,
        test_loader: DataLoader,
        dataset: SensorDataset
    ) -> Tuple[float, float]:
        """
        Evaluate model on test set.

        Args:
            test_loader: Test data loader
            dataset: Dataset object with normalization parameters

        Returns:
            mse: Mean squared error in original scale (ppm)
            r2: R² coefficient of determination
        """
        self.model.eval()
        predictions = []
        targets = []

        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                y_pred = self.model(x_batch.float())
                predictions.append(y_pred)
                targets.append(y_batch)

        predictions = torch.cat(predictions, dim=0)
        targets = torch.cat(targets, dim=0)

        # Denormalize
        predictions = SensorDataset.denormalize(
            predictions,
            dataset.y_min,
            dataset.y_max
        )
        targets = SensorDataset.denormalize(
            targets,
            dataset.y_min,
            dataset.y_max
        )

        # Compute metrics
        mse = ((predictions - targets) ** 2).mean().item()
        ss_res = ((targets - predictions) ** 2).sum()
        ss_tot = ((targets - targets.mean(dim=0)) ** 2).sum()
        r2 = (1 - ss_res / ss_tot).item()

        return mse, r2


def create_dataloaders(
    x: torch.Tensor,
    y: torch.Tensor,
    train_size: float = 0.6,
    val_size: float = 0.2,
    batch_size: int = 8,
    random_seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader, SensorDataset]:
    """
    Create train, validation, and test data loaders.

    Args:
        x: Input features
        y: Target labels
        train_size: Fraction of data for training
        val_size: Fraction of data for validation
        batch_size: Batch size
        random_seed: Random seed for reproducibility

    Returns:
        train_loader, val_loader, test_loader, train_dataset
    """
    from sklearn.model_selection import train_test_split

    # Split data
    x_train_val, x_test, y_train_val, y_test = train_test_split(
        x, y,
        test_size=1 - train_size - val_size,
        random_state=random_seed
    )

    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val, y_train_val,
        test_size=val_size / (train_size + val_size),
        random_state=random_seed
    )

    # Create datasets
    train_dataset = SensorDataset(x_train, y_train)
    val_dataset = SensorDataset(
        x_val, y_val,
        train_dataset.x_min, train_dataset.y_min,
        train_dataset.x_max, train_dataset.y_max
    )
    test_dataset = SensorDataset(
        x_test, y_test,
        train_dataset.x_min, train_dataset.y_min,
        train_dataset.x_max, train_dataset.y_max
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size * 2, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size * 2, shuffle=False)

    return train_loader, val_loader, test_loader, train_dataset
