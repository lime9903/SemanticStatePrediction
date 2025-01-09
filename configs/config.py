"""
Configuration file for main
"""
import os
import argparse

ROOT_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = os.path.join(ROOT_PATH, 'data')


def parse_arguments():
    parser = argparse.ArgumentParser(description="Semantic Extraction")

    # environment
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--data_dir_path', type=str, default=r'C:\smartthings_data\csv_data\20241128',
                        help='Path to data directory')
    parser.add_argument('--save_path', type=str, default=r'C:\Users\lime9\PycharmProjects\semanticProject',
                        help='Path to save')
    parser.add_argument('--save_vis_path', type=str,
                        default=r'C:\Users\lime9\PycharmProjects\semanticProject\visualization',
                        help='Path to save visualization')

    # model
    parser.add_argument('-b', dest='batch_size', type=int, default=50, help='minibatch size')
    parser.add_argument('-nepoch', dest='num_epochs', type=int, default=50, help='number of epochs')
    parser.add_argument('-lr', dest='learning_rate', type=float, default=1e-3, help='learning rate')

    return parser.parse_args(args=[])  # if Jupyter Notebook
    # return parser.parse_args()
