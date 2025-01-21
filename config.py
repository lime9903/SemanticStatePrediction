"""
Configuration module for the NILM project.
"""
import ast
import argparse
from pathlib import Path
from typing import Any, List
from IPython import get_ipython


class Config:
    """Configuration class for managing project settings."""

    def __init__(self):
        """Initialize configuration with default paths."""
        self.root_path = Path(__file__).parent
        self.save_path = self.root_path / 'outputs'
        self.args = self.parse_arguments()

    @staticmethod
    def arg_as_list(s: str) -> List[Any]:
        """Convert string representation of list to actual list."""
        try:
            v = ast.literal_eval(s)
            if not isinstance(v, list):
                raise argparse.ArgumentTypeError(f'Argument {s} must be a list')
            return v
        except (ValueError, SyntaxError) as e:
            raise argparse.ArgumentTypeError(f'Invalid list format: {e}')

    def parse_arguments(self) -> argparse.Namespace:
        """Parse command line arguments and return configuration."""
        parser = argparse.ArgumentParser(description="Semantic Extraction")
        self._add_environment_args(parser)   # Environmental arguments
        self._add_data_args(parser)          # Data load and processing arguments
        self._add_model_args(parser)         # Model arguments
        self._add_channel_args(parser)       # Channel arguments

        # Handle Jupyter notebook case
        try:
            get_ipython()
            return parser.parse_args(args=[])
        except:
            return parser.parse_args()

    def _add_environment_args(self, parser: argparse.ArgumentParser) -> None:
        """Add environment-related arguments."""
        env_group = parser.add_argument_group('Environment')
        env_group.add_argument(
            '--debug',
            action='store_true',
            default=True,
            help='Enable debug mode'
        )
        env_group.add_argument(
            '--device',
            type=str,
            default='cuda',
            help='Device(CPU/GPU) used for trainer'
        )
        env_group.add_argument(
            '-rand',
            dest='random_state',
            type=int,
            default=42,
            help='Random state for reproducibility'
        )
        env_group.add_argument(
            '--data_dir_path',
            type=str,
            default=self.root_path / 'data',
            help='Path to data directory'
        )
        env_group.add_argument(
            '--save_data_path',
            type=str,
            default=self.save_path / 'dataset',
            help='Path to save outputs'
        )
        env_group.add_argument(
            '--save_vis_path',
            type=str,
            default=self.save_path / 'visualization',
            help='Path to save visualization'
        )
        env_group.add_argument(
            '--save_model_path',
            type=str,
            default=self.save_path / 'checkpoints',
            help='Path to save model checkpoints'
        )

    def _add_data_args(self, parser: argparse.ArgumentParser) -> None:
        """Add data-related arguments."""
        data_group = parser.add_argument_group('Data')
        data_group.add_argument(
            '-dc', '--num_dc',
            type=int,
            choices=[1, 2, 3, 4],
            default=4,
            help='Number of data collection'
        )
        data_group.add_argument(
            '-s', '--scaler_type',
            type=str,
            choices=["standard", "minmax"],
            default="standard",
            help='Type of scaler'
        )
        data_group.add_argument(
            '-eq', '--make_equal_dist',
            action='store_true',
            default=True,
            help='Whether data distribution is equal or not'
        )
        data_group.add_argument(
            '-agg',
            dest='aggregate',
            action='store_true',
            default=True,
            help='Aggregated dataset or not'
        )

    def _add_model_args(self, parser: argparse.ArgumentParser) -> None:
        """Add model-related arguments."""
        model_group = parser.add_argument_group('Model')
        model_group.add_argument(
            '-i',
            dest='input_size',
            type=int,
            default=2,
            choices=[2, 7],
            help='Size of input for deep learning model'
        )
        model_group.add_argument(
            '-c',
            dest='num_classes',
            type=int,
            default=343,
            help='Number of classes for deep learning model'
        )
        model_group.add_argument(
            '-b',
            dest='batch_size',
            type=int,
            default=32,
            help='minibatch size'
        )
        model_group.add_argument(
            '-st',
            dest='stride',
            type=int,
            default=1,
            help='length of stride'
        )
        model_group.add_argument(
            '-nepoch',
            dest='num_epochs',
            type=int,
            default=500,
            help='number of epochs'
        )
        model_group.add_argument(
            '-lr',
            dest='learning_rate',
            type=float,
            default=1e-3,
            help='learning rate'
        )
        model_group.add_argument(
            '-p',
            dest='patience',
            type=int,
            default=15,
            help='early stopping patience'
        )
        model_group.add_argument(
            '-vs',
            dest='val_size',
            type=float,
            default=0.1,
            help='Size of validation set'
        )
        model_group.add_argument(
            '-ts',
            dest='test_size',
            type=float,
            default=0.2,
            help='Size of test set'
        )
        model_group.add_argument(
            '-seq_len',
            dest='sequence_length',
            type=int,
            default=50,
            help='Length of sequence'
        )
        model_group.add_argument(
            '--class_names',
            default=[],
            type=self.arg_as_list,
            help='List of class names'
        )
        model_group.add_argument(
            '--model_name',
            type=str,
            choices=["CNN-RNN", "CNN-LSTM", "DAE-RNN", "DAE-LSTM"],
            help='Name of the deep learning model'
        )

    def _add_channel_args(self, parser: argparse.ArgumentParser) -> None:
        """Add channel-related arguments."""
        channel_group = parser.add_argument_group('Channel')
        channel_group.add_argument(
            '--snr_db',
            type=int,
            default=20,
            help='SNR db'
        )
        channel_group.add_argument(
            '--channel_name',
            type=str,
            default="AWGN",
            choices=["AWGN", "Rayleigh", "Rician", "Nakagami"],
            help='Channel environment types'
        )
        channel_group.add_argument(
            '--doppler_freq',
            type=float,
            default=0,
            help='Doppler frequency'
        )
        channel_group.add_argument(
            '--k_factor',
            type=float,
            default=1.0,
            help='K factor of Rician channel'
        )
        channel_group.add_argument(
            '--variance',
            type=float,
            default=1.0,
            help='Variance of Rician channel'
        )
        channel_group.add_argument(
            '--m_factor',
            type=float,
            default=1.0,
            help='M factor of Nakagami channel'
        )
        channel_group.add_argument(
            '--omega',
            type=float,
            default=1.0,
            help='Average power(omega) of Nakagami channel'
        )


def get_config():
    """Get configuration instance."""
    return Config().args

# Initialize configuration
args = get_config()
