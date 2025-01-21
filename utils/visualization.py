import os
from collections import Counter
from typing import Dict, List, Tuple, Any, Union

import numpy as np
from numpy import ndarray, dtype, floating, float_
from numpy._typing import _64Bit
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams, gridspec
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import confusion_matrix

from dataloader.data_processor import StateDataProcessor

# Set style parameters
sns.set_style("whitegrid")
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']
rcParams['figure.facecolor'] = 'white'


def plot_training_history(history: Dict, save_path: str) -> None:
    """
    Plot trainer and validation metrics history.

    Args:
        history: Dictionary containing trainer history
        save_path: Path to save the plot
    """
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

    file_name = os.path.join(save_path, "training_history.png")
    plt.savefig(file_name, dpi=300)
    plt.close()
    print(f"Saved trainer history: {file_name}")


def plot_confusion_matrix(
        conf_matrix: np.ndarray,
        class_names: List[str],
        save_path: str,
        model_name: str,
        num_dc: int
) -> None:
    """
    Plot confusion matrix with percentages.

    Args:
        conf_matrix: Confusion matrix array
        class_names: List of class names
        save_path: Path to save the plot
        model_name: Name of the model
    """
    plt.figure(figsize=(25, 20))
    cmap = sns.color_palette("YlOrRd", as_cmap=True)

    row_sums = conf_matrix.sum(axis=1)
    row_sums[row_sums == 0] = 1
    norm_conf_matrix = conf_matrix.astype('float') / row_sums[:, np.newaxis]

    # Create heatmap
    ax = sns.heatmap(
        conf_matrix,
        annot=True,
        fmt='d',
        cmap=cmap,
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Number of Samples', 'orientation': 'vertical'},
        square=True
    )

    # Add percentage annotations
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            text = ax.texts[i * conf_matrix.shape[1] + j]
            if row_sums[i] == 1 and conf_matrix[i].sum() == 0:
                continue
            plt.text(
                j + 0.5,
                i + 0.7,
                f'({norm_conf_matrix[i, j]:.1%})',
                ha='center',
                va='center',
                color='black' if norm_conf_matrix[i, j] < 0.5 else 'white',
                fontsize=9
            )

    plt.title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Predicted Class', fontsize=12, fontweight='bold', labelpad=15)
    plt.ylabel('True Class', fontsize=12, fontweight='bold', labelpad=15)

    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()

    file_name = os.path.join(save_path, f"dc{num_dc}_{model_name}_confusion_matrix.png")
    plt.savefig(file_name, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion matrix: {file_name}")


def plot_model_comparison(
        histories: Dict[str, Dict],
        results: Dict[str, Dict],
        save_path: str,
        num_dc: int
) -> None:
    """
    Plot comparison of multiple models' performance.

    Args:
        histories: Dictionary of trainer histories for each model
        results: Dictionary of test results for each model
        save_path: Path to save the plots
    """
    # Plot trainer losses comparison
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

    _style_comparison_plot("Training and Validation Loss Curves")

    file_name = os.path.join(save_path, f"dc{num_dc}_model_loss_comparison.png")
    plt.savefig(file_name, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved loss comparison: {file_name}")

    # Plot test accuracies comparison
    plt.figure(figsize=(12, 7))
    model_names = list(results.keys())
    accuracies = [results[model]['test_accuracy'] for model in model_names]

    data = pd.DataFrame({
        "Model": model_names,
        "Accuracy": accuracies
    })

    # Create bar plot
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

    _style_accuracy_plot(ax, accuracies)

    file_name = os.path.join(save_path, f"dc{num_dc}_model_accuracy_comparison.png")
    plt.savefig(file_name, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved accuracy comparison: {file_name}")


def _style_comparison_plot(title: str) -> None:
    """
    Apply common styling to comparison plots.

    Args:
        title: Title of the plot
    """
    plt.grid(True, linestyle='--', alpha=0.3, which='both')
    plt.gca().set_axisbelow(True)

    # Remove top and right spines
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    # Set labels and title
    plt.xlabel('Epoch', fontsize=12, fontweight='bold', labelpad=15)
    plt.ylabel('Loss', fontsize=12, fontweight='bold', labelpad=15)
    plt.title(title, fontsize=16, fontweight='bold', pad=20, color='#2f2f2f')

    # Style legend
    legend = plt.legend(
        fontsize=11,
        frameon=True,
        facecolor='white',
        edgecolor='none',
        shadow=True,
        loc='center left',
        bbox_to_anchor=(1, 0.5)
    )
    legend.get_frame().set_alpha(0.9)

    # Set axis properties
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.gca().set_facecolor('#f8f9fa')
    plt.tight_layout()


def _style_accuracy_plot(ax: plt.Axes, accuracies: List[float]) -> None:
    """
    Apply styling to accuracy comparison plot.

    Args:
        ax: Matplotlib axes object
        accuracies: List of accuracy values
    """
    # Add grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)

    # Style spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)

    # Set labels and title
    plt.xlabel('Model', fontsize=12, fontweight='bold', labelpad=15)
    plt.ylabel('Test Accuracy', fontsize=12, fontweight='bold', labelpad=15)
    plt.title('Model Performance Comparison',
              fontsize=16,
              fontweight='bold',
              pad=20,
              color='#2f2f2f')

    # Set axis limits and ticks
    plt.ylim(0, 1.1)
    plt.xticks(rotation=45, ha='right', fontsize=11)
    plt.yticks(fontsize=11)

    # Add value annotations
    for i, acc in enumerate(accuracies):
        plt.text(i, acc + 0.02,
                 f"{acc:.3f}",
                 ha='center',
                 va='bottom',
                 fontsize=11,
                 fontweight='bold',
                 color='#2f2f2f')

    # Set background color
    ax.set_facecolor('#f8f9fa')
    plt.tight_layout()


def visualize_results(histories: Dict[str, Dict],
                      results: Dict[str, Dict],
                      args: Any) -> None:
    """
    Generate and save all visualization plots.

    Args:
        histories: Dictionary of trainer histories for each model
        results: Dictionary of test results for each model
        args: Configuration arguments
    """
    print("\nGenerating visualization plots...")

    # Plot model comparisons
    plot_model_comparison(histories, results, args.save_vis_path, args.num_dc)

    # Plot confusion matrices
    for model_name, test_result in results.items():
        conf_matrix = confusion_matrix(
            test_result['true_labels'],
            test_result['predictions']
        )
        plot_confusion_matrix(
            conf_matrix,
            args.class_names,
            args.save_vis_path,
            model_name,
            args.num_dc
        )

        print(f"\n{model_name} Test Accuracy: {test_result['test_accuracy']:.4f}")
        print(f"\n{model_name} Classification Report:")
        print(test_result['classification_report'])

    print("\nVisualization complete. Plots saved to:", args.save_vis_path)


def calculate_hamming_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculate Hamming distance between two binary vectors."""
    return np.sum(vec1 != vec2) / len(vec1)


def get_ordered_states(level_mapping: Dict[int, List[str]]) -> List[str]:
    """Get states ordered by priority level."""
    return [state for level in sorted(level_mapping.keys())
            for state in level_mapping[level]]


def analyze_embeddings(semantic_embeddings: Dict[str, np.ndarray],
                       bitwise_embeddings: Dict[str, np.ndarray],
                       level_mapping: Dict[int, List[str]],
                       args=None) -> tuple[ndarray[Any, dtype[Any]],
ndarray[Any, dtype[Union[floating[_64Bit], float_]]], ndarray[Any, dtype[Union[floating[_64Bit], float_]]]]:
    """Analyze and visualize both semantic and bit-wise embeddings."""
    ordered_states = get_ordered_states(level_mapping)
    n_states = len(ordered_states)

    # Calculate similarity matrices
    semantic_sim = np.zeros((n_states, n_states))
    hamming_dist = np.zeros((n_states, n_states))

    for i, state1 in enumerate(ordered_states):
        for j, state2 in enumerate(ordered_states):
            vec1, vec2 = semantic_embeddings[state1], semantic_embeddings[state2]
            semantic_sim[i, j] = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            hamming_dist[i, j] = calculate_hamming_distance(
                bitwise_embeddings[state1],
                bitwise_embeddings[state2]
            )

    _plot_heatmaps(semantic_sim, hamming_dist, ordered_states, level_mapping,
                   semantic_embeddings, bitwise_embeddings, args)
    _plot_correlation(semantic_sim, hamming_dist, n_states, args)

    triu_indices = np.triu_indices(n_states, k=1)
    correlation = np.corrcoef(semantic_sim[triu_indices], hamming_dist[triu_indices])[0, 1]

    return correlation, semantic_sim, hamming_dist


def _plot_heatmaps(
        semantic_sim: np.ndarray,
        hamming_dist: np.ndarray,
        ordered_states: List[str],
        level_mapping: Dict[int, List[str]],
        semantic_embeddings: Dict[str, np.ndarray],
        bitwise_embeddings: Dict[str, np.ndarray],
        args=None
) -> None:
    """Plot heatmaps for semantic and bitwise embeddings."""
    import matplotlib.gridspec as gridspec

    fig = plt.figure(figsize=(18, 10))

    # Create subplot grid with space for dimension info
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 4])

    # Add dimension information at the top
    ax_info = plt.subplot(gs[0, :])
    ax_info.axis('off')

    # Get dimensions
    bit_dim = args.num_bits if args and hasattr(args, 'num_bits') else len(next(iter(bitwise_embeddings.values())))
    sem_dim = args.embedding_dim if args and hasattr(args, 'embedding_dim') else len(
        next(iter(semantic_embeddings.values())))
    ratio = bit_dim / sem_dim if sem_dim != 0 else float('inf')

    # Create dimension info text
    info_text = (
        f"Embedding Dimensions:\n"
        f"Bitwise: {bit_dim} dimensions | "
        f"Semantic: {sem_dim} dimensions | "
        f"Ratio (Bit:Sem): {ratio:.2f}:1"
    )
    ax_info.text(0.5, 0.5, info_text,
                 horizontalalignment='center',
                 verticalalignment='center',
                 fontsize=12,
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.5'))

    # Create heatmap subplots
    ax1 = plt.subplot(gs[1, 0])
    ax2 = plt.subplot(gs[1, 1])

    # Calculate level boundaries
    level_boundaries = []
    current_pos = 0
    for level in sorted(level_mapping.keys()):
        current_pos += len(level_mapping[level])
        if current_pos < len(ordered_states):
            level_boundaries.append(current_pos)

    # Plot semantic similarity heatmap
    _plot_single_heatmap(
        ax1, semantic_sim, ordered_states, level_boundaries,
        'Semantic Embeddings (Cosine Similarity)', 'coolwarm', 'Cosine Similarity'
    )

    # Plot Hamming distance heatmap
    _plot_single_heatmap(
        ax2, hamming_dist, ordered_states, level_boundaries,
        'Bitwise Embeddings (Hamming Distance)', 'YlOrRd', 'Hamming Distance'
    )

    # Add level labels
    _add_level_labels(ax1, level_mapping)
    _add_level_labels(ax2, level_mapping)

    plt.tight_layout()
    plt.show()


def _plot_single_heatmap(
        ax: plt.Axes,
        data: np.ndarray,
        ordered_states: List[str],
        level_boundaries: List[int],
        title: str,
        cmap: str,
        cbar_label: str
) -> None:
    """Plot a single heatmap with given data and settings."""
    sns.heatmap(
        data,
        xticklabels=ordered_states,
        yticklabels=ordered_states,
        annot=True,
        fmt='.2f',
        cmap=cmap,
        ax=ax,
        cbar_kws={'label': cbar_label}
    )

    # Add level boundaries
    for boundary in level_boundaries:
        ax.hlines(boundary, *ax.get_xlim(), colors='black', linewidth=2)
        ax.vlines(boundary, *ax.get_ylim(), colors='black', linewidth=2)

    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    plt.setp(ax.get_yticklabels(), rotation=0)


def _add_level_labels(ax: plt.Axes, level_mapping: Dict[int, List[str]]) -> None:
    """Add level labels to the heatmap."""
    current_pos = 0
    for level in sorted(level_mapping.keys()):
        level_size = len(level_mapping[level])
        middle_pos = current_pos + level_size / 2
        ax.text(
            -0.15, middle_pos, f'Level {level}',
            verticalalignment='center',
            horizontalalignment='right'
        )
        current_pos += level_size


def _plot_correlation(
        semantic_sim: np.ndarray,
        hamming_sim: np.ndarray,
        n_states: int,
        args=None
) -> None:
    """
    Plot correlation between semantic similarity and binary similarity matrices.

    Args:
        semantic_sim: Matrix of semantic similarities
        hamming_sim: Matrix of binary similarities (1 - Hamming distances)
        n_states: Number of states
        args: Optional arguments containing dimension information
    """
    # Get upper triangular values (excluding diagonal)
    triu_indices = np.triu_indices(n_states, k=1)
    semantic_vals = semantic_sim[triu_indices]
    hamming_vals = hamming_sim[triu_indices]

    # Calculate correlation
    correlation = np.corrcoef(semantic_vals, hamming_vals)[0, 1]

    plt.figure(figsize=(12, 8))

    # Add dimension info at the top
    if args:
        bit_dim = args.num_bits if hasattr(args, 'num_bits') else "N/A"
        sem_dim = args.embedding_dim if hasattr(args, 'embedding_dim') else "N/A"
        plt.figtext(0.5, 0.95,
                    f'Dimensions: Bitwise={bit_dim}, Semantic={sem_dim}',
                    ha='center', va='center',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

    # Create scatter plot
    plt.scatter(semantic_vals, hamming_vals, alpha=0.5)

    # Add trend line
    z = np.polyfit(semantic_vals, hamming_vals, 1)
    p = np.poly1d(z)
    x_range = np.linspace(min(semantic_vals), max(semantic_vals), 100)
    plt.plot(
        x_range, p(x_range), "r--", alpha=0.8,
        label=f'Linear fit (y = {z[0]:.2f}x + {z[1]:.2f})'
    )

    plt.xlabel('Semantic Similarity')
    plt.ylabel('Binary Similarity (1 - Hamming Distance)')
    plt.title(f'Similarity Correlation (r = {correlation:.3f})')
    plt.grid(True)
    plt.legend()

    # Adjust layout to accommodate the dimension info
    plt.subplots_adjust(top=0.85)
    plt.show()

    # Print additional statistics
    print("\n=== Correlation Analysis ===")
    print(f"Pearson correlation coefficient: {correlation:.3f}")
    print(f"Number of state pairs analyzed: {len(semantic_vals)}")
    print("\n=== Similarity Statistics ===")
    print(f"Semantic similarity range: [{semantic_vals.min():.3f}, {semantic_vals.max():.3f}]")
    print(f"Binary similarity range: [{hamming_vals.min():.3f}, {hamming_vals.max():.3f}]")


def plot_state_distribution_pie(processor: StateDataProcessor, args) -> None:
    """
    Visualize the state distribution of train, validation, test sets using seaborn styling

    Args:
        df: Original DataFrame from DataCollectionLoader
        processor: StateDataProcessor instance
        args: Configuration arguments
    """
    sns.set_context("notebook", font_scale=1.2)
    plt.figure(figsize=(15, 5))

    datasets = {
        'Train Set': processor.y_train,
        'Validation Set': processor.y_val,
        'Test Set': processor.y_test
    }

    all_states = sorted(set(np.unique(processor.y_train)) |
                        set(np.unique(processor.y_val)) |
                        set(np.unique(processor.y_test)))

    colors = sns.color_palette("husl", n_colors=len(all_states))

    for idx, (name, data) in enumerate(datasets.items(), 1):
        plt.subplot(1, 3, idx)

        # numpy array를 위한 value counts 계산
        value_counts = Counter(data)
        values = [value_counts.get(state, 0) for state in all_states]

        wedges, texts, autotexts = plt.pie(values,
                                           labels=all_states,
                                           colors=colors,
                                           autopct=lambda pct: f'{pct:.1f}%\n({int(pct / 100. * sum(values)):d})',
                                           startangle=90)

        plt.setp(autotexts, size=8, weight="bold", color="white")
        plt.setp(texts, size=8)

        plt.title(f'{name}\nTotal Samples: {len(data)}',
                  pad=20,
                  fontsize=12,
                  fontweight='bold')

    legend = plt.figlegend(all_states,
                           title='States',
                           loc='center right',
                           bbox_to_anchor=(1.2, 0.5),
                           title_fontsize=12,
                           fontsize=10)
    legend.get_frame().set_alpha(0.9)

    plt.tight_layout()
    plt.show()
