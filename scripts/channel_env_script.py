import os

import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict

from channel.wireless_channel import AWGNChannel, RayleighChannel, RicianChannel, NakagamiChannel
from dataloader.data_loader import DataCollectionLoader
from dataloader.data_processor import StateDataProcessor
from models.embedding import SemanticBitEmbedding
from models.networks import DAE_LSTM, SemanticCNNLSTM, CNN_LSTM, SemanticDAELSTM  # TODO: get_model 변경
from config import args
from scripts.state_predict_script import prepare_dc_argument
from trainer.state_predict_trainer import train_multiple_models
from utils.visualization import visualize_results


def evaluate_semantic_transmission(model, test_loader, embedder, dc_loader, channel, args, processor):
    """
    Evaluate the entire process: state prediction -> bit embedding -> transmission -> recovery
    """

    def bits_to_signal(bit_pattern):
        return np.array([int(b) for b in format(bit_pattern, '07b')])

    def signal_to_bits(signal):
        binary_signal = np.where(signal >= 0.5, 1, 0)
        return int(''.join(str(int(b)) for b in binary_signal), 2)

    def find_nearest_pattern(received_bits, bit_embeddings):
        min_distance = float('inf')
        nearest_label = None
        hamming_dist = float('inf')

        for label, pattern in bit_embeddings.items():
            distance = bin(pattern ^ received_bits).count('1')
            if distance < min_distance:
                min_distance = distance
                nearest_label = label
                hamming_dist = distance

        return nearest_label, hamming_dist

    results = {
        'perfect_recoveries': 0,
        'hamming_errors': [],
        'semantic_errors': [],
        'total': 0,
        'error_details': []
    }

    model.eval()
    with torch.no_grad():
        for signals, true_labels in test_loader:
            signals = signals.to(args.device)
            batch_size = signals.size(0)
            results['total'] += batch_size

            outputs = model(signals)
            _, predicted_indices = torch.max(outputs.data, 1)

            predicted_labels = processor.label_encoders['state'].inverse_transform(predicted_indices.cpu().numpy())
            true_states = processor.label_encoders['state'].inverse_transform(true_labels.cpu().numpy())

            for pred_label, true_label in zip(predicted_labels, true_states):
                try:
                    original_bits = embedder.labeled_bit_embeddings[pred_label]
                    signal = bits_to_signal(original_bits)
                    # print(f'prediction label: {pred_label}')
                    # Channel Transmission
                    received = channel.apply_channel(signal)
                    if np.iscomplexobj(received):
                        received = np.abs(received)
                    if received.max() > 1 or received.min() < 0:
                        received = (received - received.min()) / (received.max() - received.min())

                    # Recovery
                    received_bits = signal_to_bits(received)
                    recovered_label, hamming_dist = find_nearest_pattern(received_bits, embedder.labeled_bit_embeddings)
                    # print(f'recovered_label: {recovered_label}')
                    if recovered_label == pred_label:
                        results['perfect_recoveries'] += 1
                    else:
                        pred_tuple = next(k for k, v in dc_loader.state_ids.items() if v == pred_label)
                        recovered_tuple = next(k for k, v in dc_loader.state_ids.items() if v == recovered_label)
                        true_tuple = next(k for k, v in dc_loader.state_ids.items() if v == true_label)

                        true_workers = sum(1 for action in true_tuple if action == 'wpc')
                        recovered_workers = sum(1 for action in recovered_tuple if action == 'wpc')
                        semantic_error = abs(true_workers - recovered_workers)

                        results['hamming_errors'].append(hamming_dist)
                        results['semantic_errors'].append(semantic_error)
                        results['error_details'].append({
                            'true_state': true_label,
                            'predicted_state': pred_label,
                            'recovered_state': recovered_label,
                            'original_bits': original_bits,
                            'received_bits': received_bits,
                            'hamming_distance': hamming_dist,
                            'semantic_error': semantic_error
                        })

                except Exception as e:
                    print(f"Error processing state {pred_label}: {str(e)}")
                    continue

    return results


def plot_semantic_transmission_analysis(all_results, save_path, channel_type):
    """
    Plot SNR vs Error Rates and Semantic Errors

    Args:
        all_results: Dictionary with SNR as key and results as value
        save_path: Path to save the plot
        channel_type: Type of channel used
    """
    plt.figure(figsize=(12, 8))

    snr_values = sorted(all_results.keys())
    error_rates = []
    semantic_errors = []

    for snr in snr_values:
        results = all_results[snr]
        error_rate = (1 - results['perfect_recoveries'] / results['total']) * 100
        error_rates.append(error_rate)

        if results['semantic_errors']:
            avg_semantic = np.mean(results['semantic_errors'])
        else:
            avg_semantic = 0
        semantic_errors.append(avg_semantic)

    fig, ax1 = plt.subplots(figsize=(12, 8))
    ax2 = ax1.twinx()

    line1 = ax1.plot(snr_values, error_rates, 'b-', marker='o', label='Error Rate', linewidth=2)
    ax1.set_xlabel('SNR (dB)', fontsize=12)
    ax1.set_ylabel('Error Rate (%)', color='b', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='b')

    line2 = ax2.plot(snr_values, semantic_errors, 'r--', marker='s', label='Avg Semantic Error', linewidth=2)
    ax2.set_ylabel('Average Semantic Error', color='r', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='r')

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    plt.legend(lines, labels, loc='upper right')

    plt.title(f'Error Analysis - {channel_type} Channel', fontsize=14, pad=20)
    plt.grid(True, alpha=0.3)

    for i, (err_rate, sem_err) in enumerate(zip(error_rates, semantic_errors)):
        ax1.annotate(f'{err_rate:.1f}%',
                     (snr_values[i], err_rate),
                     textcoords="offset points",
                     xytext=(0, 10),
                     ha='center',
                     fontsize=9)
        ax2.annotate(f'{sem_err:.2f}',
                     (snr_values[i], sem_err),
                     textcoords="offset points",
                     xytext=(0, -15),
                     ha='center',
                     fontsize=9,
                     color='r')

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"error_analysis_{channel_type}.png"))
    plt.close()


def plot_semantic_recovery_analysis(all_results, save_path, channel_type):
    """
    Plot the semantic error recovery analysis
    """
    plt.figure(figsize=(15, 10))

    # 상태 전이 매트릭스를 위한 데이터 수집
    all_transitions = []
    for snr, results in all_results.items():
        for error in results['error_details']:
            all_transitions.append((error['predicted_state'],
                                    error['recovered_state'],
                                    error['semantic_error']))

    # 유니크한 상태 목록 생성
    unique_states = sorted(list(set([t[0] for t in all_transitions] +
                                    [t[1] for t in all_transitions])))

    # 전이 매트릭스 생성
    transition_matrix = np.zeros((len(unique_states), len(unique_states)))
    semantic_matrix = np.zeros((len(unique_states), len(unique_states)))

    for pred, rec, sem_err in all_transitions:
        i = unique_states.index(pred)
        j = unique_states.index(rec)
        transition_matrix[i, j] += 1
        semantic_matrix[i, j] += sem_err

    # 평균 semantic error 계산
    with np.errstate(divide='ignore', invalid='ignore'):
        semantic_matrix = np.divide(semantic_matrix, transition_matrix,
                                    where=transition_matrix != 0)

    # 상태 전이 히트맵
    plt.subplot(1, 2, 1)
    sns.heatmap(transition_matrix,
                xticklabels=unique_states,
                yticklabels=unique_states,
                cmap='YlOrRd')
    plt.title('State Transition Frequency')
    plt.xlabel('Recovered State')
    plt.ylabel('Predicted State')

    # Semantic Error 히트맵
    plt.subplot(1, 2, 2)
    sns.heatmap(semantic_matrix,
                xticklabels=unique_states,
                yticklabels=unique_states,
                cmap='RdYlBu_r',
                mask=transition_matrix == 0)
    plt.title('Average Semantic Error')
    plt.xlabel('Recovered State')
    plt.ylabel('Predicted State')

    plt.suptitle(f'Semantic Recovery Analysis - {channel_type} Channel',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path,
                             f"semantic_recovery_analysis_{channel_type}.png"),
                bbox_inches='tight')
    plt.close()


def analyze_semantic_recovery(results, dc_loader):
    """
    Analyze the semantic recovery performance from transmission results
    """
    # 상태 간 전이와 semantic distance를 저장할 매트릭스 준비
    state_transitions = defaultdict(lambda: defaultdict(list))
    semantic_distances = []
    all_states = set()

    # True state와 recovered state 간의 관계 분석
    for error in results['error_details']:
        true_state = error['true_state']
        pred_state = error['predicted_state']
        rec_state = error['recovered_state']
        sem_error = error['semantic_error']

        all_states.add(true_state)
        all_states.add(rec_state)

        # 전이 정보 저장
        state_transitions[true_state][rec_state].append({
            'semantic_error': sem_error,
            'hamming_distance': error['hamming_distance']
        })
        semantic_distances.append(sem_error)

    return state_transitions, list(all_states), semantic_distances


def plot_recovery_pie_chart(results, save_path, channel_type, snr):
    """
    Create pie chart visualization for semantic recovery analysis

    Args:
        results: Dictionary containing evaluation results
        save_path: Path to save the visualization
        channel_type: Type of channel used
        snr: Signal-to-Noise Ratio value
    """
    plt.figure(figsize=(10, 8))

    # Calculate metrics
    perfect_recovery = results['perfect_recoveries']
    total_cases = results['total']

    # Count semantically close cases (where semantic error <= 1)
    semantic_close = sum(1 for error in results['error_details']
                         if error['semantic_error'] <= 1 and error['true_state'] != error['recovered_state'])

    # Prepare data for pie chart
    labels = ['Perfect Recovery', 'Semantically Close', 'Significant Error']
    sizes = [
        perfect_recovery,
        semantic_close,
        total_cases - (perfect_recovery + semantic_close)
    ]
    colors = ['lightgreen', 'lightyellow', 'lightcoral']
    explode = (0.1, 0.05, 0)  # 첫 번째 조각을 약간 돌출

    # Create pie chart
    plt.pie(sizes,
            explode=explode,
            labels=labels,
            colors=colors,
            autopct='%1.1f%%',
            shadow=True,
            startangle=90)

    plt.title(f'Recovery Performance Distribution\nChannel: {channel_type}, SNR: {snr}dB', pad=20)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle

    # Save plot
    plt.savefig(f'{save_path}/recovery_pie_{channel_type}_{snr}dB.png', bbox_inches='tight')
    plt.close()

    # Print summary statistics
    print(f"\nAnalysis Summary for {channel_type} Channel at {snr}dB:")
    print(f"Perfect Recovery Rate: {perfect_recovery / total_cases * 100:.2f}%")
    print(f"Semantically Close Rate: {semantic_close / total_cases * 100:.2f}%")
    print(f"Significant Error Rate: {(total_cases - perfect_recovery - semantic_close) / total_cases * 100:.2f}%")


def evaluate_with_visualization(model, test_loader, embedder, dc_loader, channel, args, processor):
    # Run evaluation
    results = evaluate_semantic_transmission(
        model, test_loader, embedder, dc_loader,
        channel, args, processor
    )

    # Create visualization
    plot_recovery_pie_chart(
        results,
        args.save_vis_path,
        channel.__class__.__name__.replace('Channel', ''),
        args.snr_db
    )

    return results


if __name__ == "__main__":
    print('[Start Execution]')

    # Configuration
    args.num_dc = 5  # TODO: enter the DC number to use - [1, 2, 3, 4]
    args.seq_len = 1000
    args.num_epochs = 500
    if args.debug:
        args.device = torch.device("cpu")

    # Load data
    dc_loader = DataCollectionLoader(args)
    processor = StateDataProcessor(args)
    df = dc_loader.load_preprocess()
    train_loader, val_loader, test_loader = processor.create_data_loaders(df)
    prepare_dc_argument(df, dc_loader, args)

    # Train and validation model
    # print("\n[Training and Testing Models]")
    # model_classes = {
    #     'Semantic-CNN-LSTM': SemanticCNNLSTM,
    #     'Semantic-DAE-LSTM': SemanticDAELSTM
    # }
    #
    # histories, results = train_multiple_models(
    #     model_classes,
    #     train_loader,
    #     val_loader,
    #     test_loader,
    #     args
    # )
    #
    # print("\n[Generating Visualizations]")
    # visualize_results(histories, results, args)

    # Create semantic bit embeddings
    embedder = SemanticBitEmbedding()
    # TODO: num_add args에 넣기
    bit_embeddings, labeled_bit_embeddings = embedder.process_states(dc_loader.states, dc_loader.state_ids, processor, num_add=3)
    if args.debug:
        print(f'bit embeddings: {bit_embeddings}')
        print(f'labeled bit embeddings: {labeled_bit_embeddings}')

    # Load the saved best model
    model_name = "dc5_best_Semantic-CNN-LSTM_model.pt"  # TODO: choose the model name to use
    model_checkpoint = os.path.join(args.save_model_path, model_name)
    model = SemanticCNNLSTM(args.input_size, args.num_classes)
    state_dict = torch.load(model_checkpoint, map_location=args.device, weights_only=True)
    model.load_state_dict(state_dict)

    if args.debug:
        print('\n')
        print(model)

    # Test transmission across different SNRs
    channel_types = ['AWGN']
    snr_range = [-5, 0, 5, 10, 15, 20, 25, 30]

    for channel_type in channel_types:
        print(f"\nTesting {channel_type} channel:")
        channel_results = {}

        for snr in snr_range:
            args.snr_db = snr

            # Create channel
            if channel_type == 'AWGN':
                channel = AWGNChannel(args)
            elif channel_type == 'Rayleigh':
                channel = RayleighChannel(args)
            elif channel_type == 'Rician':
                channel = RicianChannel(args)
            else:
                channel = NakagamiChannel(args)

            # Evaluate transmission
            results = evaluate_with_visualization(
                model, test_loader, embedder, dc_loader,
                channel, args, processor
            )

            print(f"\nSNR: {snr}dB")
            print(f"Perfect Recovery Rate: {results['perfect_recoveries'] / results['total'] * 100:.2f}%")

            if results['error_details']:
                avg_hamming = np.mean([err['hamming_distance'] for err in results['error_details']])
                avg_semantic = np.mean([err['semantic_error'] for err in results['error_details']])
                print(f"Average Hamming Distance: {avg_hamming:.3f}")
                print(f"Average Semantic Error: {avg_semantic:.3f}")

            channel_results[snr] = results
        #
        # plot_semantic_transmission_analysis(channel_results, args.save_vis_path, channel_type)
        # plot_semantic_recovery_analysis(channel_results, args.save_vis_path, channel_type)
