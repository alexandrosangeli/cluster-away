from utils import get_data, decode_encode, batched_encode, plot_drifting
import torch
from molearn.models.CNN_autoencoder import AutoEncoder as ConvolutionalAE
from molearn.models.foldingnet import AutoEncoder as FoldingNet
import argparse
import sys
import os
import json
import math
import datetime
import time


current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(current_dir, '..', '..') 
sys.path.append(root_dir)
from generic_utils.utils import AUTOENCODER_SELLECTION
from generic_utils.cli_utils import parse_all_args


def main(args):

    data_path = args['data_path']
    datafiles = args['datafiles']
    output_dir = args['output_dir']
    device = args['device']
    autoencoder_of_choice = args['autoencoder_of_choice']
    timestamp = args['timestamp']
    request_gpu = args['request_gpu']
    description = args['description']
    verbose = args['verbose']

    # Experiment specific
    num_iters = args['num_iters']
    scale_factor = args['grid_scale_factor']
    res = args['resolution']
    checkpoint_file = args['checkpoint_file']

    assert res > 1, "Resolution must be greater than 1 otherwise the code will fail"

    torch.manual_seed(2025)
    batch_size=8
    checkpoint = torch.load(checkpoint_file, map_location=torch.device('cpu'), weights_only=False)
    model = autoencoder_of_choice(**checkpoint['network_kwargs'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(device)

    data = get_data(datafiles).dataset
    data = data.to(device)
    num_atoms = data.size(1)

    initial_z = batched_encode(model=model, dataset=data, batch_size=batch_size, verbose=verbose)
    min_x = torch.min(initial_z.squeeze()[:, 0]) - (torch.min(initial_z.squeeze()[:, 0]) * scale_factor)
    min_y = torch.min(initial_z.squeeze()[:, 1]) - (torch.min(initial_z.squeeze()[:, 1]) * scale_factor)
    max_x = torch.max(initial_z.squeeze()[:, 0]) + (torch.max(initial_z.squeeze()[:, 0]) * scale_factor)
    max_y = torch.max(initial_z.squeeze()[:, 1]) + (torch.max(initial_z.squeeze()[:, 1]) * scale_factor)
    x_lin = torch.linspace(min_x, max_x, res, device=device)
    y_lin = torch.linspace(min_y, max_y, res, device=device)
    X, Y = torch.meshgrid(x_lin, y_lin, indexing='xy')

    startings = torch.stack([X.flatten(), Y.flatten()], dim=1)
    endings = decode_encode(model=model, z=startings, num_iters=num_iters, num_atoms=num_atoms, batch_size=batch_size, verbose=verbose)
    
    torch.save(endings['encodings'][-1, :, :], f"{output_dir}/{timestamp}_encodings.pt")
    print(f"Saved encodings in {output_dir}/{timestamp}_encodings.pt")

    plot_drifting(z=endings['encodings'], num_iters=num_iters, output_dir=output_dir, res=res, timestamp=timestamp, min_x=None, max_x=max_x, min_y=min_y, max_y=max_y, gif=True)

    print("Script complete. Exiting.")
    return 0

if __name__ == "__main__":
    start_time = time.time()
    args = parse_all_args(description="Drift experiment arg parser", experiment=sys.argv[0])
    main(args)
    end_time = time.time()
    duration_seconds = end_time - start_time
    minutes = math.floor(duration_seconds / 60) 
    remaining_seconds = duration_seconds % 60
    print(f"Python script duration: **{minutes} minutes and {remaining_seconds:.2f} seconds**")