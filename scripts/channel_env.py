import os

import numpy as np
import torch

from channel.wireless_channel import AWGNChannel, RayleighChannel
from models.hybrid import CNN_RNN
from configs.config import parse_arguments


if __name__ == "__main__":
    # Configuration
    args = parse_arguments()
    args.num_dc = 4   # TODO: number of DC
    args.input_size = 2
    args.num_classes = 16
    snr_db = 20  # TODO: argparse에 넣기

    if args.debug:
        args.device = torch.device("cpu")

    # Test data


    # Load saved best model
    model_name = "dc4_best_CNN_RNN_model.pt"
    model_checkpoint = os.path.join(args.save_model_path, model_name)
    model = CNN_RNN(args.input_size, args.num_classes)
    state_dict = torch.load(model_checkpoint, map_location=args.device, weights_only=True)
    model.load_state_dict(state_dict)
    print(model)

    awgn_channel = AWGNChannel('AWGN Channel', snr_db)
    rayleigh_channel = RayleighChannel('Rayleigh Channel', snr_db)

    input_signal = np.array([1, -1, 1, -1])  #
    output_signal = awgn_channel.apply_channel(input_signal)
    print(output_signal)