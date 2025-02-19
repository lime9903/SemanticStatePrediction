"""
Main script for trainer and evaluating models.
"""
import torch
from pathlib import Path

from config import args
from dataloader.data_loader import DataCollectionLoader
from dataloader.data_processor import StateDataProcessor
from models.networks import *
from trainer.state_predict_trainer import train_multiple_models
from utils.visualization import visualize_results, plot_state_distribution_pie


def prepare_dc_argument(df, dc_loader, args):
    """Prepare data collection specific arguments."""
    assert args.num_dc in [1, 2, 3, 4, 5], "Invalid data collection number"

    # Set input size based on aggregation
    args.input_size = 2 if args.aggregate else 7

    # Set output size and class names
    unique_states = sorted(df['state'].unique().tolist())  # Get actual states present in data
    args.num_classes = len(unique_states)
    args.class_names = unique_states  # Use only the states that actually appear in the data

    print(f"\nActual number of classes in data: {args.num_classes}")
    print(f"Number of states present in data: {len(unique_states)}")
    print(f"States present in data: {unique_states}")


def setup_device(args):
    """Setup computation device."""
    if args.debug:
        return torch.device("cpu")

    if args.device == 'cuda':
        if not torch.cuda.is_available():
            print("WARNING: CUDA device not available, using CPU.")
            return torch.device('cpu')
        return torch.device('cuda')

    return torch.device('cpu')


def print_debug_info(args, dc_loader, processor):
    """Print debug information."""
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
    print('X_val.shape:', processor.X_val.shape)
    print('X_test.shape:', processor.X_test.shape)
    print('y_train.shape:', processor.y_train.shape)
    print('y_val.shape:', processor.y_val.shape)
    print('y_test.shape:', processor.y_test.shape)

    print("\n[Data Format]")
    print('X_train:\n', processor.X_train)


def main():
    """Main execution function."""
    print('[Start Execution]')
    args.num_dc = 5   # Enter your DC number
    args.seq_len = 15
    args.num_epochs = 500

    # Setup device
    args.device = setup_device(args)

    # Create output directories
    Path(args.save_model_path).mkdir(parents=True, exist_ok=True)
    Path(args.save_vis_path).mkdir(parents=True, exist_ok=True)

    # Data preparation and preprocessing
    dc_loader = DataCollectionLoader(args)
    processor = StateDataProcessor(args)
    df = dc_loader.load_preprocess()
    train_loader, val_loader, test_loader = processor.create_data_loaders(df)
    prepare_dc_argument(df, dc_loader, args)

    # Print debug information if needed
    if args.debug:
        print_debug_info(args, dc_loader, processor)

    # Define models to train
    model_names = ['CNN-RNN', 'CNN-LSTM', 'Semantic-CNN-LSTM']
    # ['CNN-RNN', 'CNN-LSTM']
    # ['DAE-RNN','DAE-LSTM']
    # ['Semantic-CNN-LSTM', 'Semantic-DAE-LSTM']

    # Train and test models
    print("\n[Training and Testing Models]")
    histories, results = train_multiple_models(
        model_names,
        train_loader,
        val_loader,
        test_loader,
        args
    )

    # Visualize results
    print("\n[Generating Visualizations]")
    visualize_results(histories, results, args)


if __name__ == '__main__':
    main()
