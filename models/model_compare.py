import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from torch import optim

from models.hybrid import *
from scripts.train import ModelTrainer


class ModelComparison:
    def __init__(self, args):
        self.args = args
        self.models = {
            'CNN-RNN': CNN_RNN(args.input_size, args.num_classes),
            'CNN-LSTM': CNN_LSTM(args.input_size, args.num_classes),
            'DAE-RNN': DAE_RNN(args.input_size, args.num_classes),
            'DAE-LSTM': DAE_LSTM(args.input_size, args.num_classes)
        }
        self.histories = {}
        self.test_results = {}

    def train_all_models(self, train_loader, val_loader, test_loader):
        for model_name, model in self.models.items():
            print(f"\n=== Training {model_name} ===")
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters())

            trainer = ModelTrainer(model, criterion, optimizer, self.args)
            self.histories[model_name] = trainer.train(train_loader, val_loader)
            self.test_results[model_name] = trainer.test(test_loader)

    def plot_combined_training_history(self, save_path: str):
        """Plot training histories of all models in a single figure"""
        plt.figure(figsize=(15, 10))

        # Create subplot for losses
        plt.subplot(2, 1, 1)
        for model_name in self.models.keys():
            plt.plot(self.histories[model_name]['train_losses'],
                     label=f'{model_name} Train Loss', linestyle='--', alpha=0.7)
            plt.plot(self.histories[model_name]['val_losses'],
                     label=f'{model_name} Val Loss', alpha=0.7)

        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Losses Comparison')
        plt.legend()
        plt.grid(True)

        # Create subplot for validation accuracies
        plt.subplot(2, 1, 2)
        for model_name in self.models.keys():
            plt.plot(self.histories[model_name]['val_accuracies'],
                     label=f'{model_name}', alpha=0.7)

        plt.xlabel('Epoch')
        plt.ylabel('Validation Accuracy')
        plt.title('Validation Accuracy Comparison')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        file_name = os.path.join(save_path, f"dc{self.args.num_dc}_model_comparison.png")
        plt.savefig(file_name, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved comparison graph: {file_name}")

    def plot_final_metrics_comparison(self, save_path: str):
        """Plot bar charts comparing final metrics across models"""
        metrics = {
            'Test Accuracy': [],
            'Training Time': [],
            'Best Val Loss': []
        }

        model_names = list(self.models.keys())

        for model_name in model_names:
            metrics['Test Accuracy'].append(self.test_results[model_name]['test_accuracy'])
            metrics['Training Time'].append(self.histories[model_name]['training_time'])
            metrics['Best Val Loss'].append(self.histories[model_name]['best_val_loss'])

        fig, axes = plt.subplots(1, 3, figsize=(20, 6))

        # Plot test accuracy
        axes[0].bar(model_names, metrics['Test Accuracy'])
        axes[0].set_title('Test Accuracy')
        axes[0].set_ylabel('Accuracy')
        axes[0].tick_params(axis='x', rotation=45)

        # Plot training time
        axes[1].bar(model_names, metrics['Training Time'])
        axes[1].set_title('Training Time')
        axes[1].set_ylabel('Seconds')
        axes[1].tick_params(axis='x', rotation=45)

        # Plot best validation loss
        axes[2].bar(model_names, metrics['Best Val Loss'])
        axes[2].set_title('Best Validation Loss')
        axes[2].set_ylabel('Loss')
        axes[2].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        file_name = os.path.join(save_path, f"dc{self.args.num_dc}_final_metrics_comparison.png")
        plt.savefig(file_name, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved final metrics comparison: {file_name}")

    def plot_confusion_matrices(self, save_path: str):
        """Plot confusion matrices for all models in a single figure"""
        fig, axes = plt.subplots(2, 2, figsize=(20, 20))
        axes = axes.ravel()

        for idx, (model_name, results) in enumerate(self.test_results.items()):
            conf_matrix = confusion_matrix(results['true_labels'], results['predictions'])

            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                        xticklabels=self.args.class_names,
                        yticklabels=self.args.class_names,
                        ax=axes[idx])

            axes[idx].set_title(f'{model_name} Confusion Matrix')
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_ylabel('True')
            axes[idx].tick_params(axis='both', rotation=45)

        plt.tight_layout()
        file_name = os.path.join(save_path, f"dc{self.args.num_dc}_all_confusion_matrices.png")
        plt.savefig(file_name, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved confusion matrices: {file_name}")

    def print_comparative_report(self):
        """Print a comparative report of all models"""
        print("\n=== Model Comparison Report ===")

        # Create comparison table
        comparison_data = []
        for model_name in self.models.keys():
            model_metrics = {
                'Model': model_name,
                'Test Accuracy': f"{self.test_results[model_name]['test_accuracy']:.4f}",
                'Training Time': f"{self.histories[model_name]['training_time']:.2f}s",
                'Best Val Loss': f"{self.histories[model_name]['best_val_loss']:.4f}"
            }
            comparison_data.append(model_metrics)

        df = pd.DataFrame(comparison_data)
        print("\nPerformance Metrics:")
        print(df.to_string(index=False))

        # Print detailed classification metrics
        print("\nDetailed Classification Metrics:")
        for model_name in self.models.keys():
            print(f"\n{model_name}:")
            report = self.test_results[model_name]['classification_report']
            metrics_df = pd.DataFrame(report).transpose()
            print(metrics_df.round(4))


def run_model_comparison(args, train_loader, val_loader, test_loader):
    """Main function to run model comparison"""
    comparison = ModelComparison(args)

    # Train all models
    comparison.train_all_models(train_loader, val_loader, test_loader)

    # Generate visualizations
    comparison.plot_combined_training_history(args.save_vis_path)
    comparison.plot_final_metrics_comparison(args.save_vis_path)
    comparison.plot_confusion_matrices(args.save_vis_path)

    # Print comparative report
    comparison.print_comparative_report()

    return comparison
