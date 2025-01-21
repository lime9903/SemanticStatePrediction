"""
Model trainer and evaluation module.
"""
import time
from pathlib import Path
from typing import Dict, Tuple, Any, Union, Type

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

from models.networks import get_model


class ModelTrainer:
    """Trainer class for model trainer and evaluation."""

    def __init__(self, model: nn.Module, args: Any):
        """
        Initialize trainer with model and configuration.

        Args:
            model: Neural network model to train
            args: Training configuration arguments
        """
        self.model = model.to(args.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters())
        self.args = args
        self.device = args.device

        # Track metrics
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []

        # Setup save directory
        self.save_dir = Path(args.save_model_path)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """
        Train the model for one epoch.

        Args:
            train_loader: Training data loader

        Returns:
            Tuple of (epoch_loss, epoch_accuracy)
        """
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(X_batch)
            loss = self.criterion(outputs, y_batch)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Track metrics
            total_loss += loss.item()
            predictions = outputs.argmax(dim=1)
            correct += (predictions == y_batch).sum().item()
            total += y_batch.size(0)

        return total_loss / len(train_loader), correct / total

    def evaluate(self, test_loader: DataLoader) -> Tuple[float, float, np.ndarray, np.ndarray]:
        """
        Evaluate the model on test data.

        Args:
            test_loader: Test data loader

        Returns:
            Tuple of (average_loss, accuracy, predictions, true_labels)
        """
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)

                # Track metrics
                total_loss += loss.item()
                predictions = outputs.argmax(dim=1)
                correct += (predictions == y_batch).sum().item()
                total += y_batch.size(0)

                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())

        avg_loss = total_loss / len(test_loader)
        accuracy = correct / total

        return avg_loss, accuracy, np.array(all_predictions), np.array(all_labels)

    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict:
        """
        Train the model with early stopping.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader

        Returns:
            Dictionary containing trainer history
        """
        best_val_loss = float('inf')
        patience_counter = 0
        start_time = time.time()
        best_model_path = self.save_dir / f'dc{self.args.num_dc}_best_{self.args.model_name}_model.pt'

        for epoch in range(self.args.num_epochs):
            # Train one epoch
            train_loss, train_accuracy = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)

            # Evaluate on validation set
            val_loss, val_accuracy, _, _ = self.evaluate(val_loader)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_accuracy)

            # Print progress
            if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == self.args.num_epochs - 1:
                print(f"\nEpoch {epoch + 1}/{self.args.num_epochs}")
                print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
                print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), best_model_path)
                print(f"Saved improved model at epoch {epoch + 1} with validation loss {val_loss:.4f}")
            else:
                patience_counter += 1

            if patience_counter >= self.args.patience:
                print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                break

        training_time = time.time() - start_time
        print(f"\nTotal trainer time: {training_time:.2f} seconds")

        # Load best model
        state_dict = torch.load(best_model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)

        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'best_val_loss': best_val_loss,
            'training_time': training_time,
        }

    def test(self, test_loader: DataLoader) -> Dict:
        """
        Test the model and compute metrics.

        Args:
            test_loader: Test data loader

        Returns:
            Dictionary containing test results and metrics
        """
        test_loss, test_accuracy, predictions, true_labels = self.evaluate(test_loader)
        report = classification_report(
            true_labels,
            predictions,
            target_names=self.args.class_names,
            output_dict=True,
            zero_division=0
        )

        return {
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'classification_report': report,
            'predictions': predictions,
            'true_labels': true_labels
        }


def train_and_evaluate_model(
        model_class: nn.Module,
        model_name: str,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        args: Any
) -> Tuple[Dict, Dict]:
    """
    Train, validate, and test a single model.

    Args:
        model_class: Model class to instantiate
        model_name: Name of the model
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader
        args: Configuration arguments

    Returns:
        Tuple of (training_history, test_results)
    """
    print(f"\n[{model_name} Training]")
    args.model_name = model_name
    model = get_model(args.model_name, args.input_size, args.num_classes)
    trainer = ModelTrainer(model, args)

    history = trainer.train(train_loader, val_loader)

    print(f"\n[{model_name} Testing]")
    test_results = trainer.test(test_loader)
    print("\nTest Accuracy:", test_results['test_accuracy'])

    return history, test_results


def train_multiple_models(
        model_classes: Dict[str, nn.Module],
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        args: Any
) -> Tuple[Dict[str, Dict], Dict[str, Dict]]:
    """
    Train and evaluate multiple models.

    Args:
        model_classes: Dictionary mapping model names to model classes
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader
        args: Configuration arguments

    Returns:
        Tuple of (training_histories, test_results)
    """
    histories = {}
    results = {}

    for model_name, model_class in model_classes.items():
        history, test_results = train_and_evaluate_model(
            model_class, model_name, train_loader, val_loader, test_loader, args
        )
        histories[model_name] = history
        results[model_name] = test_results

    return histories, results
