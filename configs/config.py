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
    parser.add_argument('--data_dir_path', type=str, default=r'C:\Users\lime9\PycharmProjects\semanticProject\data',
                        help='Path to data directory')
    parser.add_argument('--save_out_path', type=str, default=r'C:\Users\lime9\PycharmProjects\semanticProject\outputs',
                        help='Path to save outputs')
    parser.add_argument('--save_vis_path', type=str,
                        default=r"C:\Users\lime9\PycharmProjects\semanticProject\visualization",
                        help='Path to save visualization')

    # data
    parser.add_argument('--num_dc', type=int, default=4, help='Number of data collection, Available:[1, 2, 3, 4]')

    # model
    parser.add_argument('-b', dest='batch_size', type=int, default=50, help='minibatch size')
    parser.add_argument('-nepoch', dest='num_epochs', type=int, default=50, help='number of epochs')
    parser.add_argument('-lr', dest='learning_rate', type=float, default=1e-3, help='learning rate')

    return parser.parse_args(args=[])  # if Jupyter Notebook
    # return parser.parse_args()
