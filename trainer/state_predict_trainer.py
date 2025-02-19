"""
Model trainer and evaluation module.
"""
import time
from pathlib import Path
from typing import Dict, Tuple, Any, Union, Type, List
from abc import ABC, abstractmethod

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

from models.networks import create_model


class BaseTrainer(ABC):
    """Base trainer class with common functionalities"""
    def __init__(self, model: nn.Module, args: Any):
        self.model = model.to(args.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters())
        self.args = args
        self.device = args.device

        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []

        self.save_dir = Path(args.save_model_path)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def evaluate(self, val_loader: DataLoader) -> Tuple[float, float, np.ndarray, np.ndarray]:
        """Evaluate model on validation data."""
        pass

    def test(self, test_loader: DataLoader) -> Dict:
        """Test the model and compute metrics."""
        test_loss, test_accuracy, predictions, true_labels = self.evaluate(test_loader)
        report = classification_report(
            true_labels,
            predictions,
            labels=list(range(len(self.args.class_names))),
            target_names=self.args.class_names,
            output_dict=True,
            zero_division=0
        )

        unique_predictions = sorted(set(predictions))
        unique_true_labels = sorted(set(true_labels))

        return {
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'classification_report': report,
            'predictions': predictions,
            'true_labels': true_labels,
            'num_predicted_classes': len(unique_predictions),
            'num_true_classes': len(unique_true_labels),
            'predicted_classes': unique_predictions,
            'true_classes': unique_true_labels
        }


class ConvTrainer(BaseTrainer):
    """Trainer for standard classification models (CNN-RNN, CNN-LSTM, Semantic-CNN-LSTM)"""
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(X_batch)
            loss = self.criterion(outputs, y_batch)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            predictions = outputs.argmax(dim=1)
            correct += (predictions == y_batch).sum().item()
            total += y_batch.size(0)

        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total

        return avg_loss, accuracy

    def evaluate(self, test_loader: DataLoader) -> Tuple[float, float, np.ndarray, np.ndarray]:
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

                total_loss += loss.item()
                predictions = outputs.argmax(dim=1)
                correct += (predictions == y_batch).sum().item()
                total += y_batch.size(0)

                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())

        return (total_loss / len(test_loader), correct / total,
                np.array(all_predictions), np.array(all_labels))

    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict:
        """Train the model with early stopping."""
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


class AutoEncoderTrainer(BaseTrainer):
    """Trainer for DAE-based models (DAE-RNN, DAE-LSTM, Semantic-DAE-LSTM)"""

    def __init__(self, model: nn.Module, args: Any):
        super().__init__(model, args)
        self.reconstruction_criterion = nn.MSELoss()
        self.recon_weight = 0.1

    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float, float, float]:
        self.model.train()
        total_loss = 0
        total_cls_loss = 0
        total_recon_loss = 0
        correct = 0
        total = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

            self.optimizer.zero_grad()
            cls_output, recon_output = self.model(X_batch)

            cls_loss = self.criterion(cls_output, y_batch)
            recon_loss = self.reconstruction_criterion(recon_output, X_batch)

            loss = cls_loss + self.recon_weight * recon_loss

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total_cls_loss += cls_loss.item()
            total_recon_loss += recon_loss.item()
            predictions = cls_output.argmax(dim=1)
            correct += (predictions == y_batch).sum().item()
            total += y_batch.size(0)

        avg_loss = total_loss / len(train_loader)
        avg_cls_loss = total_cls_loss / len(train_loader)
        avg_recon_loss = total_recon_loss / len(train_loader)
        accuracy = correct / total

        return avg_loss, accuracy, avg_cls_loss, avg_recon_loss

    def evaluate(self, test_loader: DataLoader) -> Tuple[float, float, np.ndarray, np.ndarray]:
        self.model.eval()
        total_loss = 0
        total_cls_loss = 0
        total_recon_loss = 0
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

                cls_output, recon_output = self.model(X_batch)

                cls_loss = self.criterion(cls_output, y_batch)
                recon_loss = self.reconstruction_criterion(recon_output, X_batch)
                loss = cls_loss + self.recon_weight * recon_loss

                total_loss += loss.item()
                total_cls_loss += cls_loss.item()
                total_recon_loss += recon_loss.item()

                predictions = cls_output.argmax(dim=1)
                correct += (predictions == y_batch).sum().item()
                total += y_batch.size(0)

                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())

        return (total_loss / len(test_loader), correct / total,
                np.array(all_predictions), np.array(all_labels))

    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict:
        best_val_loss = float('inf')
        patience_counter = 0
        start_time = time.time()
        best_model_path = self.save_dir / f'dc{self.args.num_dc}_best_{self.args.model_name}_model.pt'

        for epoch in range(self.args.num_epochs):
            # Train one epoch
            train_loss, train_acc, train_cls_loss, train_recon_loss = self.train_epoch(train_loader)
            val_loss, val_acc, _, _ = self.evaluate(val_loader)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"\nEpoch {epoch + 1}/{self.args.num_epochs}")
                print(f"Train - Total Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
                print(f"      - Class Loss: {train_cls_loss:.4f}, Recon Loss: {train_recon_loss:.4f}")
                print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), best_model_path)
                print(f"Saved model at epoch {epoch + 1} with val loss {val_loss:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= self.args.patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

        training_time = time.time() - start_time
        print(f"\nTotal trainer time: {training_time:.2f} seconds")

        state_dict = torch.load(best_model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)

        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'best_val_loss': best_val_loss,
            'training_time': training_time,
        }


def train_and_evaluate_model(
        model_name: str,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        args: Any
) -> Tuple[Dict, Dict]:
    """
    Train, validate, and test a single model.

    Args:
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
    model = create_model(args.model_name, args.input_size, args.num_classes)

    if 'DAE' in model_name:
        trainer = AutoEncoderTrainer(model, args)
    else:
        trainer = ConvTrainer(model, args)

    history = trainer.train(train_loader, val_loader)
    test_results = trainer.test(test_loader)

    print(f"\n[{model_name} Testing]")
    print("\nTest Accuracy:", test_results['test_accuracy'])

    return history, test_results


def train_multiple_models(
        model_names: List[str],
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        args: Any
) -> Tuple[Dict[str, Dict], Dict[str, Dict]]:
    """Train and evaluate multiple models."""
    histories = {}
    results = {}

    for model_name in model_names:
        history, test_results = train_and_evaluate_model(
            model_name, train_loader, val_loader, test_loader, args
        )
        histories[model_name] = history
        results[model_name] = test_results

    return histories, results
