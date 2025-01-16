import os

import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, List
from matplotlib import rcParams

from channel.wireless_channel import AWGNChannel, RayleighChannel, RicianChannel, NakagamiChannel
from dataloader.preprocess_dataset import DataCollectionLoader
from dataloader.state_dataset import StateDataProcessor
from models.hybrid import CNN_RNN, DAE_LSTM
from configs.config import parse_arguments
from scripts.train import prepare_dc_argument


# TODO: definition of channel is wired !!REVISE
def apply_channel_effects(test_loader, channel, device):
    """
    Apply channel effects to the test data with proper complex number handling
    Args:
        test_loader: DataLoader containing test data
        channel: Channel object that applies the channel effects
        device: torch device to use
    Returns:
        modified_test_loader: DataLoader with channel effects applied
    """
    modified_data = []
    original_labels = []

    # Iterate through the test loader
    for signals, labels in test_loader:
        signals_np = signals.numpy()
        modified_signals = []

        for signal in signals_np:
            original_shape = signal.shape
            flattened = signal.reshape(-1)

            processed = channel.apply_channel(flattened)

            if np.iscomplexobj(processed):
                # Option 1: Use magnitude
                # processed = np.abs(processed)
                # Option 2: Use real part only
                processed = np.real(processed)

            processed = processed.reshape(original_shape)
            modified_signals.append(processed)

        modified_signals = np.array(modified_signals)
        modified_signals = torch.from_numpy(modified_signals).float().to(device)

        modified_data.append(modified_signals)
        original_labels.append(labels)

    # Concatenate all batches
    X_test_modified = torch.cat(modified_data)
    y_test = torch.cat(original_labels)

    print(X_test_modified)
    print(test_loader.dataset[0])

    # Create new dataset and dataloader
    modified_dataset = torch.utils.data.TensorDataset(X_test_modified, y_test)
    modified_test_loader = torch.utils.data.DataLoader(
        modified_dataset,
        batch_size=test_loader.batch_size,
        shuffle=False
    )

    return modified_test_loader


def plot_snr_comparison(snr_results: Dict[str, Dict[float, float]], save_path: str):
    """
    Plot test accuracies across different SNR values for different channels/models
    """
    sns.set_style("whitegrid")
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = ['Arial']

    plt.figure(figsize=(12, 7))

    colors = sns.color_palette("husl", len(snr_results))

    for idx, (name, results) in enumerate(snr_results.items()):
        snr_values = list(results.keys())
        accuracies = list(results.values())

        plt.plot(snr_values, accuracies,
                 label=name,
                 color=colors[idx],
                 linewidth=2.5,
                 marker='o',
                 markersize=8,
                 alpha=0.9)

        for snr, acc in zip(snr_values, accuracies):
            plt.text(snr, acc + 0.01, f'{acc:.3f}',
                     ha='center',
                     va='bottom',
                     fontsize=9,
                     color=colors[idx],
                     fontweight='bold')

    plt.xlabel('SNR (dB)', fontsize=12, fontweight='bold', labelpad=15)
    plt.ylabel('Test Accuracy', fontsize=12, fontweight='bold', labelpad=15)
    plt.title('Test Accuracy vs SNR Comparison',
              fontsize=16,
              fontweight='bold',
              pad=20,
              color='#2f2f2f')

    plt.grid(True, linestyle='--', alpha=0.7)
    plt.gca().set_axisbelow(True)

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    legend = plt.legend(fontsize=11,
                        frameon=True,
                        facecolor='white',
                        edgecolor='none',
                        shadow=True,
                        loc='lower right')
    legend.get_frame().set_alpha(0.9)

    plt.yticks(fontsize=10)
    plt.xticks(fontsize=10)
    plt.tight_layout()

    file_name = os.path.join(save_path, "snr_accuracy_comparison.png")
    plt.savefig(file_name, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved SNR comparison graph: {file_name}")


def test_across_snr(model, test_loader, channel_types: List[str], snr_range: List[float], args):
    """
    Test model across different SNR values and channel types
    """
    assert all(channel_type in ["AWGN", "Rayleigh", "Rician", "Nakagami"] for channel_type in channel_types), \
        "Unsupported channel type"

    results = {}

    for channel_type in channel_types:
        results[channel_type] = {}

        for snr in snr_range:
            if channel_type == 'AWGN':
                channel = AWGNChannel(args)
            elif channel_type == 'Rayleigh':
                channel = RayleighChannel(args)
            elif channel_type == 'Rician':
                channel = RicianChannel(args)
            else:
                channel = NakagamiChannel(args)

            test_loader_with_channel = apply_channel_effects(test_loader, channel, args.device)

            model.eval()
            correct = 0
            total = 0

            with torch.no_grad():
                for signals, labels in test_loader_with_channel:
                    signals = signals.to(args.device)
                    labels = labels.to(args.device)

                    outputs = model(signals)
                    _, predicted = torch.max(outputs.data, 1)

                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            accuracy = correct / total
            results[channel_type][snr] = accuracy

            print(f"Channel: {channel_type}, SNR: {snr}dB, Accuracy: {accuracy:.4f}")

    return results


if __name__ == "__main__":
    print('[Start Execution]')

    # Configuration
    args = parse_arguments()
    args.num_dc = 4  # TODO: enter the DC number to use - [1, 2, 3, 4]
    args.input_size = 2
    args.num_classes = 16
    args.snr_db = 9
    args.channel_name = 'AWGN'  # TODO: choose the channel name to use - ["AWGN", "Rayleigh", "Rician", "Nakagami"]

    if args.debug:
        args.device = torch.device("cpu")

    # Load data
    dc_loader = DataCollectionLoader(args)
    processor = StateDataProcessor(args)
    df = dc_loader.load_preprocess()
    train_loader, val_loader, test_loader = processor.create_data_loaders(df)
    prepare_dc_argument(df, dc_loader, args)

    # Load saved best model
    model_name = "dc4_best_DAE-LSTM_model.pt"  # TODO: choose the model name to use
    model_checkpoint = os.path.join(args.save_model_path, model_name)
    model = DAE_LSTM(args.input_size, args.num_classes)
    state_dict = torch.load(model_checkpoint, map_location=args.device, weights_only=True)
    model.load_state_dict(state_dict)

    if args.debug:
        print(model)

    channel_types = ['AWGN', 'Rayleigh', 'Rician', 'Nakagami']
    snr_range = [-5, 0, 5, 10, 15, 20, 25, 30]

    snr_results = test_across_snr(model, test_loader, channel_types, snr_range, args)

    plot_snr_comparison(snr_results, args.save_vis_path)
