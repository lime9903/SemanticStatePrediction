import itertools
import os
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Any, Union, Optional, Type

import numpy as np
from numpy import ndarray, dtype, floating, float_
from numpy._typing import _64Bit
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import confusion_matrix
from scipy.spatial.distance import squareform

import torch
from torchviz import make_dot
from torchview import draw_graph
from config import args


os.environ["PATH"] += os.pathsep+'C:/Program Files/Graphviz/bin/'


def set_style():
    """Set the default style for all plots."""
    sns.set_theme(style="whitegrid", palette="husl")
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial']


def plot_training_history(history: Dict, save_path: str) -> None:
    """
    Plot trainer and validation metrics history.

    Args:
        history: Dictionary containing trainer history
        save_path: Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

    ax1.plot(history['train_losses'], label='Train Loss')
    ax1.plot(history['val_losses'], label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)

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
    set_style()
    plt.figure(figsize=(25, 20))
    cmap = sns.color_palette("YlOrRd", as_cmap=True)

    row_sums = conf_matrix.sum(axis=1)
    row_sums[row_sums == 0] = 1
    norm_conf_matrix = conf_matrix.astype('float') / row_sums[:, np.newaxis]

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
    set_style()
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

    plt.figure(figsize=(12, 7))
    model_names = list(results.keys())
    accuracies = [results[model]['test_accuracy'] for model in model_names]

    data = pd.DataFrame({
        "Model": model_names,
        "Accuracy": accuracies
    })

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

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    plt.xlabel('Epoch', fontsize=12, fontweight='bold', labelpad=15)
    plt.ylabel('Loss', fontsize=12, fontweight='bold', labelpad=15)
    plt.title(title, fontsize=16, fontweight='bold', pad=20, color='#2f2f2f')

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
    set_style()
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
        plt.text(i, acc + 0.02,
                 f"{acc:.3f}",
                 ha='center',
                 va='bottom',
                 fontsize=11,
                 fontweight='bold',
                 color='#2f2f2f')

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

    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 4])

    ax_info = plt.subplot(gs[0, :])
    ax_info.axis('off')

    bit_dim = args.num_bits if args and hasattr(args, 'num_bits') else len(next(iter(bitwise_embeddings.values())))
    sem_dim = args.embedding_dim if args and hasattr(args, 'embedding_dim') else len(
        next(iter(semantic_embeddings.values())))
    ratio = bit_dim / sem_dim if sem_dim != 0 else float('inf')

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

    ax1 = plt.subplot(gs[1, 0])
    ax2 = plt.subplot(gs[1, 1])

    level_boundaries = []
    current_pos = 0
    for level in sorted(level_mapping.keys()):
        current_pos += len(level_mapping[level])
        if current_pos < len(ordered_states):
            level_boundaries.append(current_pos)

    _plot_single_heatmap(
        ax1, semantic_sim, ordered_states, level_boundaries,
        'Semantic Embeddings (Cosine Similarity)', 'coolwarm', 'Cosine Similarity'
    )

    _plot_single_heatmap(
        ax2, hamming_dist, ordered_states, level_boundaries,
        'Bitwise Embeddings (Hamming Distance)', 'YlOrRd', 'Hamming Distance'
    )

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
    set_style()
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
    triu_indices = np.triu_indices(n_states, k=1)
    semantic_vals = semantic_sim[triu_indices]
    hamming_vals = hamming_sim[triu_indices]

    correlation = np.corrcoef(semantic_vals, hamming_vals)[0, 1]
    plt.figure(figsize=(12, 8))

    if args:
        bit_dim = args.num_bits if hasattr(args, 'num_bits') else "N/A"
        sem_dim = args.embedding_dim if hasattr(args, 'embedding_dim') else "N/A"
        plt.figtext(0.5, 0.95,
                    f'Dimensions: Bitwise={bit_dim}, Semantic={sem_dim}',
                    ha='center', va='center',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

    plt.scatter(semantic_vals, hamming_vals, alpha=0.5)

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
    plt.subplots_adjust(top=0.85)
    plt.show()

    print("\n=== Correlation Analysis ===")
    print(f"Pearson correlation coefficient: {correlation:.3f}")
    print(f"Number of state pairs analyzed: {len(semantic_vals)}")
    print("\n=== Similarity Statistics ===")
    print(f"Semantic similarity range: [{semantic_vals.min():.3f}, {semantic_vals.max():.3f}]")
    print(f"Binary similarity range: [{hamming_vals.min():.3f}, {hamming_vals.max():.3f}]")


def plot_state_distribution_pie(y_train, y_val, y_test, args, save_dir = None) -> None:
    """Visualize the state distribution of train, validation, test sets with pie chart."""
    set_style()
    sns.set_context("notebook", font_scale=1.0)
    plt.figure(figsize=(15, 7))

    datasets = {
        'Train Set': y_train,
        'Validation Set': y_val,
        'Test Set': y_test
    }

    all_states = sorted(set(np.unique(y_train)) |
                        set(np.unique(y_val)) |
                        set(np.unique(y_test)))

    colors = sns.color_palette("husl", n_colors=len(all_states))

    for idx, (name, data) in enumerate(datasets.items(), 1):
        plt.subplot(1, 3, idx)
        value_counts = Counter(data)
        values = [value_counts.get(state, 0) for state in all_states]

        wedges, texts, autotexts = plt.pie(values,
                                           labels=all_states,
                                           colors=colors,
                                           autopct=lambda pct: f'{pct:.1f}%\n({int(pct / 100. * sum(values)):d})',
                                           startangle=90)

        plt.setp(autotexts, size=7, weight="bold", color="white")
        plt.setp(texts, size=7)

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

    if save_dir is None:
        save_dir = args.save_vis_path
    save_path = os.path.join(save_dir, f"dc{args.num_dc}_state_distribution_pie.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Plot saved to {save_path}")

    plt.show()


def set_graphviz_fonts():
    """Set default fonts for Graphviz"""
    import graphviz
    graphviz.set_default_engine('dot')
    return {
        'fontname': 'Arial',
        'fontsize': '12',
        'rankdir': 'TB',
    }


def visualize_model_structure(model: torch.nn.Module,
                              input_size: int,
                              sequence_length: int = 50,
                              save_dir: str = None) -> None:
    """
    Visualize the model structure using both torchviz and torchview
    """
    if save_dir is None:
        save_dir = args.save_vis_path
    os.makedirs(save_dir, exist_ok=True)

    x = torch.randn(1, sequence_length, input_size)

    # Method 1: torchviz
    try:
        output = model(x) if not hasattr(model, 'is_dae') else model(x)[0]
        dot = make_dot(output, params=dict(model.named_parameters()))

        dot.attr('graph', rankdir='TB')
        dot.attr('node',
                 shape='box',
                 style='rounded,filled',
                 fillcolor='lightblue',
                 fontsize='10')
        dot.attr('edge',
                 arrowsize='0.5',
                 fontsize='10')

        viz_path = os.path.join(save_dir, f"{model.__class__.__name__}_torchviz")
        dot.render(viz_path, format='png', cleanup=True)
        print(f"✓ Torchviz visualization saved: {viz_path}.png")

    except Exception as e:
        print(f"✗ Torchviz visualization failed: {str(e)}")

    # Method 2: torchview
    try:
        model_graph = draw_graph(
            model,
            input_size=(1, sequence_length, input_size),
            expand_nested=True,
            graph_name=f"{model.__class__.__name__} Architecture",
            save_graph=True,
            filename=os.path.join(save_dir, f"{model.__class__.__name__}_torchview"),
            directory=save_dir
        )
        print(f"✓ Torchview visualization saved: {os.path.join(save_dir, model.__class__.__name__)}_torchview.png")

    except Exception as e:
        print(f"✗ Torchview visualization failed: {str(e)}")


class BitEmbeddingVisualizer:
    def __init__(self, bit_embeddings, dc_loader, scores=None, adjusted_scores=None):
        self.bit_embeddings = bit_embeddings
        self.dc_loader = dc_loader
        self.scores = scores  # 기본 점수
        self.adjusted_scores = adjusted_scores  # 빈도 조정된 점수
        self.num_bits = max(int.bit_length(v) for v in bit_embeddings.values())

    def get_binary_strings(self):
        """Convert integer embeddings to binary strings with proper padding"""
        binary_dict = {}
        for state, value in self.bit_embeddings.items():
            binary = bin(value)[2:].zfill(self.num_bits)
            binary_dict[state] = binary
        return binary_dict

    def print_bit_embeddings_by_score(self):
        """Print bit embeddings sorted by scores"""
        binary_dict = self.get_binary_strings()

        embedding_data = []
        for state in binary_dict.keys():
            data = {
                'state': state,
                'binary': binary_dict[state],
                'decimal': self.bit_embeddings[state],
                'base_score': self.scores[state] if self.scores else None,
                'adjusted_score': self.adjusted_scores[state] if self.adjusted_scores else None
            }
            embedding_data.append(data)

        if self.adjusted_scores:
            embedding_data.sort(key=lambda x: (-x['adjusted_score'], -x['base_score']))

        print("\nBit Embeddings (Sorted by Score):")
        print("-" * 80)
        header = f"{'State':40} {'Binary':15} {'Decimal':8}"
        if self.scores:
            header += f" {'Base Score':12}"
        if self.adjusted_scores:
            header += f" {'Adj Score':12}"
        print(header)
        print("-" * 80)

        for data in embedding_data:
            state_str = ' | '.join(data['state'])
            line = f"{state_str:40} {data['binary']:15} {data['decimal']:<8}"
            if self.scores:
                line += f" {data['base_score']:<12.5f}"
            if self.adjusted_scores:
                line += f" {data['adjusted_score']:<12.5f}"
            print(line)

    def plot_bit_matrix_by_score(self):
        """Plot the bit matrix sorted by scores with borderlines"""
        binary_dict = self.get_binary_strings()

        states = list(binary_dict.keys())
        if self.adjusted_scores:
            states.sort(key=lambda x: (-self.adjusted_scores[x], -self.scores[x]))

        bit_matrix = np.array([[int(bit) for bit in binary_dict[state]]
                               for state in states])

        plt.figure(figsize=(12, 10))
        sns.heatmap(bit_matrix, cmap='binary', cbar=False,
                    xticklabels=[f'Bit {i}' for i in range(self.num_bits)],
                    yticklabels=[f"{' | '.join(state)} (S:{self.adjusted_scores[state]:.2f})"
                                 if self.adjusted_scores else ' | '.join(state)
                                 for state in states],
                    linewidths=1,
                    linecolor='black')

        plt.gca().spines['top'].set_visible(True)
        plt.gca().spines['right'].set_visible(True)
        plt.gca().spines['bottom'].set_visible(True)
        plt.gca().spines['left'].set_visible(True)

        plt.title('Bit Embedding Matrix (Sorted by Score)')
        plt.xlabel('Bit Position')
        plt.ylabel('State')
        plt.tight_layout()
        plt.show()

    def plot_hamming_distances(self):
        """Plot histogram of Hamming distances between states"""
        binary_dict = self.get_binary_strings()
        binary_values = list(binary_dict.values())

        distances = []
        labels = []
        for i in range(len(binary_values)):
            for j in range(i + 1, len(binary_values)):
                dist = sum(a != b for a, b in zip(binary_values[i], binary_values[j]))
                distances.append(dist)
                labels.append((list(binary_dict.keys())[i], list(binary_dict.keys())[j]))

        plt.figure(figsize=(10, 6))
        plt.hist(distances, bins=range(min(distances), max(distances) + 2, 1),
                 rwidth=0.8, align='left')
        plt.title('Distribution of Hamming Distances')
        plt.xlabel('Hamming Distance')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        plt.show()

        print("\nHamming Distance Statistics:")
        print(f"Average Distance: {np.mean(distances):.2f}")
        print(f"Min Distance: {min(distances)}")
        print(f"Max Distance: {max(distances)}")

    def plot_semantic_clusters(self):
        """Plot semantic clusters based on Hamming distances"""
        binary_dict = self.get_binary_strings()
        states = list(binary_dict.keys())

        if self.adjusted_scores:
            states.sort(key=lambda x: (-self.adjusted_scores[x], -self.scores[x]))

        distances = []
        for i in range(len(states)):
            for j in range(i + 1, len(states)):
                bin1 = binary_dict[states[i]]
                bin2 = binary_dict[states[j]]
                dist = sum(a != b for a, b in zip(bin1, bin2))
                distances.append(dist)
        distance_matrix = squareform(distances)

        plt.figure(figsize=(12, 10))
        sns.heatmap(distance_matrix,
                    xticklabels=[f"{' | '.join(state)} (S:{self.adjusted_scores[state]:.2f})"
                                 if self.adjusted_scores else ' | '.join(state)
                                 for state in states],
                    yticklabels=[f"{' | '.join(state)} (S:{self.adjusted_scores[state]:.2f})"
                                 if self.adjusted_scores else ' | '.join(state)
                                 for state in states],
                    cmap='viridis')
        plt.title('Hamming Distance Matrix')

        plt.tight_layout()
        plt.show()

    def plot_distance_correlation(self):
        """
        Plot correlation between Hamming distances of bit embeddings
        and adjusted scores differences.
        """
        def calculate_hamming_distance(bits1, bits2):
            return sum(b1 != b2 for b1, b2 in zip(bits1, bits2))

        binary_dict = self.get_binary_strings()
        state_pairs = list(itertools.combinations(self.bit_embeddings.keys(), 2))

        distances = [(
            calculate_hamming_distance(binary_dict[s1], binary_dict[s2]),
            abs(self.adjusted_scores[s1] - self.adjusted_scores[s2])
        ) for s1, s2 in state_pairs]

        hamming_distances, score_differences = zip(*distances)

        plt.figure(figsize=(10, 8))
        correlation = np.corrcoef(hamming_distances, score_differences)[0, 1]

        plt.scatter(hamming_distances, score_differences, alpha=0.5)
        z = np.polyfit(hamming_distances, score_differences, 1)
        x_range = np.linspace(min(hamming_distances), max(hamming_distances), 100)
        plt.plot(x_range, np.poly1d(z)(x_range), "r--", alpha=0.8)

        plt.xlabel('Hamming Distance')
        plt.ylabel('Adjusted Score Difference')
        plt.title(f'Correlation between Hamming Distance and Score Difference\n(Correlation: {correlation:.3f})')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(sorted(set(hamming_distances)))
        plt.yticks(sorted(set(score_differences)))

        plt.tight_layout()
        plt.show()

    def visualize_all(self):
        """Generate all visualizations"""
        self.print_bit_embeddings_by_score()
        self.plot_bit_matrix_by_score()
        self.plot_hamming_distances()
        self.plot_semantic_clusters()
        self.plot_distance_correlation()


def visualize_aggregated_power(df: pd.DataFrame, save_dir=None):
    """
    Visualize the aggregated power.
    """
    set_style()
    plt.figure(figsize=(25, 6))

    sns.lineplot(data=df, x=pd.to_datetime(df['Timestamp']),
                 y='Aggregated Power', linewidth=1, color='royalblue')

    plt.title('Aggregated Power Over Time', fontsize=16, pad=20, fontweight='bold')
    plt.xlabel('Timestamp', fontsize=12, labelpad=10)
    plt.ylabel('Aggregated Power (W)', fontsize=12, labelpad=10)

    plt.grid(True, linestyle='--', alpha=0.4)
    plt.gcf().autofmt_xdate()
    plt.gca().set_facecolor('#f8f9fa')

    plt.tight_layout()

    if save_dir is None:
        save_dir = args.save_vis_path
    save_path = os.path.join(save_dir, f"dc{args.num_dc}_aggregated_power.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Plot saved to {save_path}")

    plt.show()


def visualize_dataloaders(train_dataset, val_dataset, test_dataset, save_dir=None):
    set_style()
    fig, axs = plt.subplots(3, 1, figsize=(20, 12))

    datasets = [
        (train_dataset.features[:, :, 1].numpy(), train_dataset.labels.numpy(), 'Training Data'),
        (val_dataset.features[:, :, 1].numpy(), val_dataset.labels.numpy(), 'Validation Data'),
        (test_dataset.features[:, :, 1].numpy(), test_dataset.labels.numpy(), 'Test Data')
    ]

    all_states = np.unique(np.concatenate([
        train_dataset.labels.numpy(),
        val_dataset.labels.numpy(),
        test_dataset.labels.numpy()
    ]))

    max_length = max(len(features.flatten()) for features, _, _ in datasets)

    # color setting for better visualization
    base_colors1 = plt.cm.tab20(np.linspace(0, 1, 20))
    base_colors2 = plt.cm.tab20b(np.linspace(0, 1, 20))
    base_colors3 = plt.cm.tab20c(np.linspace(0, 1, 20))
    base_colors4 = plt.cm.Set3(np.linspace(0, 1, 12))
    base_colors5 = plt.cm.Paired(np.linspace(0, 1, 12))
    colors = np.vstack([base_colors1, base_colors2, base_colors3, base_colors4, base_colors5])
    colors = colors[:len(all_states)]

    for (features, labels, title), ax in zip(datasets, axs):
        features_flat = features.flatten()

        non_zero_indices = np.where(~np.isnan(features_flat))[0]
        if len(non_zero_indices) > 0:
            start_idx = non_zero_indices[0]
            end_idx = non_zero_indices[-1] + 1
            features_flat = features_flat[start_idx:end_idx]
            time_steps = np.arange(len(features_flat))
        else:
            time_steps = np.arange(len(features_flat))

        sns.lineplot(x=time_steps, y=features_flat, ax=ax, color='black', alpha=0.9, linewidth=0.5)

        state_changes = [0]
        current_state = labels[0]
        seq_len = features.shape[1]

        for i in range(1, len(labels)):
            if labels[i] != current_state:
                state_changes.append(i * seq_len)
                current_state = labels[i]
        state_changes.append(len(features_flat))

        # state background color and number
        for i in range(len(state_changes) - 1):
            start = state_changes[i]
            end = state_changes[i + 1]
            state = labels[start // seq_len]

            # state color
            ax.axvspan(start, end, alpha=0.3, color=colors[int(state)],
                       label=f'State {state}' if i == 0 or labels[start // seq_len] != labels[
                           (start - 1) // seq_len] else "")

            # state label
            center = (start + end) / 2
            y_pos = ax.get_ylim()[1]
            ax.text(center, y_pos * 0.9, f'{int(state)}',
                    horizontalalignment='center',
                    verticalalignment='top',
                    fontsize=8)
        ax.set_xlim(0, max_length)

        ax.set_title(title, fontsize=14, pad=15, fontweight='bold')
        ax.set_xlabel('Time Steps', fontsize=11, labelpad=10)
        ax.set_ylabel('Aggregated Power (Scaled)', fontsize=11, labelpad=10)

        ax.grid(True, linestyle='--', alpha=0.4, zorder=0)
        ax.set_facecolor('white')

    plt.tight_layout()

    if save_dir is None:
        save_dir = args.save_vis_path
    save_path = os.path.join(save_dir, f"dc{args.num_dc}_datasets_visualization.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\nPlot saved to {save_path}")

    plt.show()


def plot_state_distribution(train_dataset, val_dataset, test_dataset, save_dir=None):
    """
    Plot the distribution of states in each dataset.

    Args:
        train_dataset: Training dataset (StateDataset)
        val_dataset: Validation dataset (StateDataset)
        test_dataset: Test dataset (StateDataset)
        save_dir: Directory to save the plot
    """
    set_style()

    train_labels = train_dataset.labels.numpy()
    val_labels = val_dataset.labels.numpy()
    test_labels = test_dataset.labels.numpy()

    all_data = []
    for labels, name in zip([train_labels, val_labels, test_labels],
                            ['Train', 'Validation', 'Test']):
        unique, counts = np.unique(labels, return_counts=True)
        percentages = counts / len(labels) * 100
        for state, percentage in zip(unique, percentages):
            all_data.append({
                'Dataset': name,
                'State': state,
                'Percentage': percentage
            })

    df_dist = pd.DataFrame(all_data)

    plt.figure(figsize=(20, 7))

    sns.barplot(data=df_dist, x='State', y='Percentage', hue='Dataset',
                palette='husl', alpha=0.8)

    plt.title('State Distribution Across Datasets',
              fontsize=16, pad=20, fontweight='bold')
    plt.xlabel('State', fontsize=12, labelpad=10)
    plt.ylabel('Percentage (%)', fontsize=12, labelpad=10)

    plt.legend(title='Dataset', title_fontsize=12,
               bbox_to_anchor=(1.05, 1), loc='upper left',
               frameon=True, fancybox=True, shadow=True)

    # ax = plt.gca()
    # for container in ax.containers:
    #     ax.bar_label(container, fmt='%.1f%%', padding=3)

    plt.gca().set_facecolor('#f8f9fa')
    plt.tight_layout()

    if save_dir is None:
        save_dir = args.save_vis_path
    save_path = os.path.join(save_dir, f"dc{args.num_dc}_state_distribution.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Plot saved to {save_path}")

    plt.show()
