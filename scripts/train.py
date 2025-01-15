import os
import time
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Tuple, List

import torch
from torch import nn
from torch import optim
from sklearn.metrics import classification_report, confusion_matrix

from configs.config import parse_arguments
from models.hybrid import CNN_RNN, CNN_LSTM, DAE_RNN, DAE_LSTM
from dataloader.preprocess_dataset import DataCollectionLoader
from dataloader.state_dataset import StateDataProcessor


def prepare_dc_argument():
    assert args.num_dc in [1, 2, 3, 4]

    # check input size
    if args.aggregate:
        args.input_size = 2
    else:
        args.input_size = 7

    # check output size
    args.num_classes = len(dc_loader.states)

    # check class names
    args.class_names = df['state'].unique().tolist()
    return


def plot_training_history(history: Dict, save_path: str):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

    # Plot losses
    ax1.plot(history['train_losses'], label='Train Loss')
    ax1.plot(history['val_losses'], label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)

    # Plot accuracy
    ax2.plot(history['val_accuracies'], label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Validation Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()

    file_name = os.path.join(save_path, f"dc{args.num_dc}_loss_history.png")
    plt.savefig(file_name, dpi=300)
    plt.close()
    print(f"Saved graph: {file_name}")
    return fig


def plot_confusion_matrix(conf_matrix: np.ndarray, class_names: List[str], save_path: str):
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, cbar_kws={'label': 'Counts'})
    plt.title('Confusion Matrix', fontsize=14)
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)

    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)

    file_name = os.path.join(save_path, f"dc{args.num_dc}_confusion_matrix.png")
    plt.savefig(file_name, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved graph: {file_name}")
    return plt


class ModelTrainer:
    def __init__(self, model, criterion, optimizer, args):
        self.model = model.to(args.device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.args = args
        self.device = args.device
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []

        self.save_dir = Path(args.save_model_path)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def train_epoch(self, train_loader) -> Tuple[float, float]:
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

        epoch_loss = total_loss / len(train_loader)
        epoch_accuracy = correct / total
        return epoch_loss, epoch_accuracy

    def evaluate(self, test_loader) -> Tuple[float, float, np.ndarray, np.ndarray]:
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        correct = 0
        total = 0

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

        avg_loss = total_loss / len(test_loader)
        accuracy = correct / total
        return avg_loss, accuracy, np.array(all_predictions), np.array(all_labels)

    def train(self, train_loader, val_loader) -> Dict:
        best_val_loss = float('inf')
        patience_counter = 0
        start_time = time.time()
        best_model_path = self.save_dir / f'dc{args.num_dc}_best_model.pt'

        for epoch in range(self.args.num_epochs):
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

            train_loss = total_loss / len(train_loader)
            train_accuracy = correct / total
            self.train_losses.append(train_loss)

            val_loss, val_accuracy, _, _ = self.evaluate(val_loader)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_accuracy)

            if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == self.args.num_epochs - 1:
                print(f"\nEpoch {epoch + 1}/{self.args.num_epochs}")
                print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
                print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

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
        print(f"\nTotal training time: {training_time:.2f} seconds")

        # Load the best model for later evaluation
        state_dict = torch.load(best_model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)

        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'best_val_loss': best_val_loss,
            'training_time': training_time
        }

    def test(self, test_loader) -> Dict:
        test_loss, test_accuracy, predictions, true_labels = self.evaluate(test_loader)

        report = classification_report(true_labels, predictions,
                                       target_names=args.class_names,
                                       output_dict=True)

        return {
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'classification_report': report,
            'predictions': predictions,
            'true_labels': true_labels
        }


# TODO: train과 test의 class distribution 출력하기
# Debug Example
if __name__ == '__main__':
    print('[Start Execution]')
    # Configuration
    args = parse_arguments()
    args.num_dc = 2  # TODO: choice [1, 2, 3, 4] depends on your dc choice

    if args.debug:
        args.device = torch.device("cpu")

    if args.device == 'cuda':
        if not torch.cuda.is_available():
            print("WARNING: You don't have a CUDA device, so run with CPU.")
            device = torch.device('cpu')
        else:
            device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Data prepare and preprocessing
    dc_loader = DataCollectionLoader(args)
    processor = StateDataProcessor(args)
    df = dc_loader.load_preprocess()
    train_loader, test_loader, train_loader_prac, test_loader_prac = processor.create_data_loaders(df)
    prepare_dc_argument()

    if args.debug:
        print("\n[Arguments Configuration]")
        print(args)

        print("\n[Problem Setting]")
        print("DC Number:", dc_loader.num_dc)
        print("Appliances Mapping:", dc_loader.appliances_mapping)
        print("Activity Mapping:", dc_loader.activities_mapping)
        print("Actions:", dc_loader.actions)
        print("Number of Actions:", len(dc_loader.actions))
        print("Users:", dc_loader.users)
        print("Number of Users:", dc_loader.num_users)
        print("States:", dc_loader.states)
        print("Number of States:", len(dc_loader.states))
        print("State indices:", dc_loader.state_ids)
        print("Time Interval:", dc_loader.time_interval)
        print("Input Size:", args.input_size)
        print("Output Size:", args.num_classes)
        print("Class names:", args.class_names)

        print("\n[Data Shape]")
        print('X_train.shape:', processor.X_train.shape)
        print('X_test.shape:', processor.X_test.shape)
        print('y_train.shape:', processor.y_train.shape)
        print('y_test.shape:', processor.y_test.shape)
        print('X_train_prac.shape:', processor.X_train_prac.shape)
        print('X_test_prac.shape:', processor.X_test_prac.shape)

        print("\n[Practical Data Format]")
        print('X_train_prac:\n', processor.X_train_prac)

    # =======================================================================================
    # TODO: change the model you want
    model = CNN_RNN(args.input_size, args.num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    # =======================================================================================

    if args.debug:
        print("\n[Model Configuration]")
        print(f'Model Structure:\n{model}')
        print(f'Criterion:\n{criterion}')
        print(f'Optimizer:\n{optimizer}')

    # Training
    print("\n[Training the model]")
    trainer = ModelTrainer(model, criterion, optimizer, args)
    history = trainer.train(train_loader_prac, test_loader_prac)

    # Test
    print("\n[Evaluating the model]")
    test_results = trainer.test(test_loader_prac)

    # Visualization
    plot_training_history(history, args.save_vis_path)
    conf_matrix = confusion_matrix(test_results['true_labels'], test_results['predictions'])
    plot_confusion_matrix(conf_matrix, args.class_names, args.save_vis_path)

    if args.debug:
        print("\nTest Accuracy:", test_results['test_accuracy'])
        print("\nClassification Report:")
        print(test_results['classification_report'])
