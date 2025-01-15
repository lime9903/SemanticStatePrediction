"""
Configuration file for main
"""
import os
import ast
import argparse
from IPython import get_ipython

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
SAVE_PATH = os.path.join(ROOT_PATH, 'outputs')


def arg_as_list(s):
    v = ast.literal_eval(s)
    if type(v) is not list:
        raise argparse.ArgumentTypeError(f'Argument {s} must be a list')
    return v


def parse_arguments():
    parser = argparse.ArgumentParser(description="Semantic Extraction")

    # environment
    parser.add_argument('--debug', action='store_true', default=True, help='Enable debug mode')
    parser.add_argument('--device', type=str, default='cuda', help='Device(CPU/GPU) used for training')
    parser.add_argument('-rand', dest='random_state', type=int, default=42, help='Random state for reproducibility')
    parser.add_argument('--data_dir_path', type=str, default=os.path.join(ROOT_PATH, 'data'),
                        help='Path to data directory')
    parser.add_argument('--save_data_path', type=str, default=os.path.join(SAVE_PATH, 'dataset'),
                        help='Path to save outputs')
    parser.add_argument('--save_vis_path', type=str, default=os.path.join(SAVE_PATH, 'visualization'),
                        help='Path to save visualization')
    parser.add_argument('--save_model_path', type=str, default=os.path.join(SAVE_PATH, 'checkpoints'),
                        help='Path to save model checkpoints')

    # data
    parser.add_argument('-dc', '--num_dc', type=int, choices=[1, 2, 3, 4], default=4,
                        help='Number of data collection')
    parser.add_argument('-s', '--scaler_type', type=str, choices=["standard", "minmax"],
                        default="standard", help='Type of scaler')
    parser.add_argument('-eq', '--make_equal_dist', action='store_true', default=True,
                        help='Whether data distribution is equal or not')
    parser.add_argument('-agg', dest='aggregate', action='store_true', default=True,
                        help='Aggregated dataset or not')

    # model
    parser.add_argument('-i', dest='input_size', type=int, default=2, choices=[2, 7],
                        help='Size of input for deep learning model')
    parser.add_argument('-c', dest='num_classes', type=int, default=343,
                        help='Number of classes for deep learning model')
    parser.add_argument('-b', dest='batch_size', type=int, default=32, help='minibatch size')
    parser.add_argument('-nepoch', dest='num_epochs', type=int, default=50, help='number of epochs')
    parser.add_argument('-lr', dest='learning_rate', type=float, default=1e-3, help='learning rate')
    parser.add_argument('-p', dest='patience', type=int, default=10, help='early stopping patience')
    parser.add_argument('-ts', dest='test_size', type=float, default=0.2, help='Size of test set')
    parser.add_argument('-seq_len', dest='sequence_length', type=int, default=32, help='Length of sequence')
    parser.add_argument('--class_names', default=[], type=arg_as_list, help='List of class names')

    # Check if running in Jupyter notebook
    try:
        get_ipython()
        return parser.parse_args(args=[])
    except:
        return parser.parse_args()
