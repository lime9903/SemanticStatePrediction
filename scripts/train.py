import os
import time
from pathlib import Path
from typing import Dict, Tuple, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib import rcParams
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from torch import optim

from configs.config import parse_arguments
from dataloader.preprocess_dataset import DataCollectionLoader
from dataloader.state_dataset import StateDataProcessor
from models.hybrid import CNN_RNN, CNN_LSTM, DAE_RNN, DAE_LSTM

sns.set_style("whitegrid")
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']
rcParams['figure.facecolor'] = 'white'


def prepare_dc_argument(df, dc_loader, args):
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


def plot_test_accuracies(results: Dict[str, Dict], save_path: str):
    """
    Plot the test accuracies of different models
    """
    plt.figure(figsize=(12, 7))
    model_names = list(results.keys())
    accuracies = [results[model]['test_accuracy'] for model in model_names]
    data = pd.DataFrame({"Model": model_names, "Accuracy": accuracies})
    base_colors = sns.color_palette("husl", len(model_names))
    transparent_colors = [(r, g, b, 0.75) for r, g, b in base_colors]

    ax = sns.barplot(
        data=data,
        x="Model",
        y="Accuracy",
        hue="Model",
        palette=transparent_colors,
        legend=False
    )

    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)

    plt.xlabel('Model', fontsize=12, fontweight='bold', labelpad=15)
    plt.ylabel('Test Accuracy', fontsize=12, fontweight='bold', labelpad=15)
    plt.title('Model Performance Comparison',
              fontsize=16,
              fontweight='bold',
              pad=20,
              color='#2f2f2f')

    plt.ylim(0, 1.1)
    plt.xticks(rotation=45, ha='right', fontsize=11)
    plt.yticks(fontsize=11)

    for i, acc in enumerate(accuracies):
        plt.text(i, acc + 0.02, f"{acc:.3f}",
                 ha='center',
                 va='bottom',
                 fontsize=11,
                 fontweight='bold',
                 color='#2f2f2f')

    ax.set_facecolor('#f8f9fa')
    plt.tight_layout()

    file_name = os.path.join(save_path, "test_accuracies_comparison.png")
    plt.savefig(file_name, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved accuracy comparison graph: {file_name}")


def plot_confusion_matrix(conf_matrix: np.ndarray, class_names: List[str], file_name: str):
    plt.figure(figsize=(12, 10))
    cmap = sns.color_palette("YlOrRd", as_cmap=True)
    norm_conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

    ax = sns.heatmap(conf_matrix,
                     annot=True,
                     fmt='d',
                     cmap=cmap,
                     xticklabels=class_names,
                     yticklabels=class_names,
                     cbar_kws={'label': 'Number of Samples',
                               'orientation': 'vertical'},
                     square=True)

    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            text = ax.texts[i * conf_matrix.shape[1] + j]
            plt.text(j + 0.5, i + 0.7, f'({norm_conf_matrix[i, j]:.1%})',
                     ha='center', va='center',
                     color='black' if norm_conf_matrix[i, j] < 0.5 else 'white',
                     fontsize=9)

    plt.title('Confusion Matrix',
              fontsize=16,
              fontweight='bold',
              pad=20,
              color='#2f2f2f')
    plt.xlabel('Predicted Class', fontsize=12, fontweight='bold', labelpad=15)
    plt.ylabel('True Class', fontsize=12, fontweight='bold', labelpad=15)

    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()

    plt.savefig(file_name, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion matrix: {file_name}")


def plot_training_losses(histories: Dict[str, Dict], save_path: str):
    """
    Plot enhanced training and validation losses
    """
    plt.figure(figsize=(14, 8))
    colors = sns.color_palette("husl", len(histories))

    for idx, (model_name, history) in enumerate(histories.items()):
        plt.plot(history['train_losses'],
                 label=f'{model_name} (Train)',
                 color=colors[idx],
                 linewidth=2.5,
                 alpha=0.9)
        plt.plot(history['val_losses'],
                 label=f'{model_name} (Val)',
                 linestyle='--',
                 color=colors[idx],
                 linewidth=2,
                 alpha=0.9)

    plt.grid(True, linestyle='--', alpha=0.3, which='both')
    plt.gca().set_axisbelow(True)

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    plt.xlabel('Epoch', fontsize=12, fontweight='bold', labelpad=15)
    plt.ylabel('Loss', fontsize=12, fontweight='bold', labelpad=15)
    plt.title('Training and Validation Loss Curves',
              fontsize=16,
              fontweight='bold',
              pad=20,
              color='#2f2f2f')

    legend = plt.legend(fontsize=11,
                        frameon=True,
                        facecolor='white',
                        edgecolor='none',
                        shadow=True,
                        loc='center left',
                        bbox_to_anchor=(1, 0.5))
    legend.get_frame().set_alpha(0.9)

    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.gca().set_facecolor('#f8f9fa')
    plt.tight_layout()

    file_name = os.path.join(save_path, "training_losses_comparison.png")
    plt.savefig(file_name, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved loss comparison graph: {file_name}")


class ModelTrainer:
    def __init__(self, model, args):
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
        best_model_path = self.save_dir / f'dc{args.num_dc}_best_{args.model_name}_model.pt'

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
            'training_time': training_time,
        }

    def test(self, test_loader) -> Dict:
        test_loss, test_accuracy, predictions, true_labels = self.evaluate(test_loader)
        report = classification_report(true_labels, predictions,
                                       target_names=args.class_names,
                                       output_dict=True,
                                       zero_division=0)

        return {
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'classification_report': report,
            'predictions': predictions,
            'true_labels': true_labels
        }


def train_and_evaluate_model(model_class, model_name, train_loader, val_loader, test_loader, args) -> Tuple[
    Dict, Dict]:
    """
    Train, validate, and test a model.
    """
    print(f"\n[{model_name} Training]")
    args.model_name = model_name
    model = model_class(args.input_size, args.num_classes)
    trainer = ModelTrainer(model, args)

    # Train the model
    history = trainer.train(train_loader, val_loader)

    # Test the model
    print(f"\n[{model_name} Testing]")
    test_results = trainer.test(test_loader)
    print("\nTest Accuracy:", test_results['test_accuracy'])

    return history, test_results


def train_multiple_models(model_classes: Dict[str, object], train_loader, val_loader, test_loader, args) -> Tuple[
    Dict, Dict]:
    """
    Train and evaluate multiple models.
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


def visualize_results(histories: Dict, results: Dict, args):
    """
    Visualize training histories and test results.
    """
    plot_training_losses(histories, args.save_vis_path)
    plot_test_accuracies(results, args.save_vis_path)

    for model_name, test_result in results.items():
        file_name = os.path.join(args.save_vis_path, f'dc{args.num_dc}_{model_name}_confusion_matrix.png')
        conf_matrix = confusion_matrix(test_result['true_labels'], test_result['predictions'])
        plot_confusion_matrix(conf_matrix, args.class_names, file_name)

        print(f"\n{model_name} Test Accuracy: {test_result['test_accuracy']}")
        print(f"\n{model_name} Classification Report:")
        print(test_result['classification_report'])


# TODO: train과 test의 class distribution 출력
if __name__ == '__main__':
    print('[Start Execution]')

    # Configuration
    args = parse_arguments()
    args.num_dc = 4  # TODO: choice [1, 2, 3, 4] depends on your dc choice

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
    train_loader, val_loader, test_loader = processor.create_data_loaders(df)
    prepare_dc_argument(df, dc_loader, args)

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

        print("\n[Data Format]")
        print('X_train:\n', processor.X_train)

    # Define models
    model_classes = {
        'CNN-RNN': CNN_RNN,
        'CNN-LSTM': CNN_LSTM,
        'DAE-RNN': DAE_RNN,
        'DAE-LSTM': DAE_LSTM
    }

    # Train and Test
    print("\n[Train/Test the Model]")
    histories, results = train_multiple_models(model_classes, train_loader, val_loader, test_loader, args)

    # Visualization
    print("\n[Visualization: Loss comparison, accuracy comparison, confusion matrix]")
    visualize_results(histories, results, args)
