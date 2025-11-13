from utils import get_data, decode_encode, batched_encode, plot_drifting
import torch
from molearn.models.CNN_autoencoder import AutoEncoder as ConvolutionalAE
from molearn.models.foldingnet import AutoEncoder as FoldingNet
import argparse
import sys
import json
import math
import datetime
import time


AUTOENCODER_SELLECTION = {
    "cnn_ae" : ConvolutionalAE,
    "fold_net" : FoldingNet
}


def log_params(path, **params):
    print("--- Parameters ---")
    print(json.dumps(params, indent=4))
    print("------------------")
    with open(f"{path}/params.json", 'w') as f:
            json.dump(params, f, indent=4)
    print(f"Parameters successfully written to: {path}")


def main():
    parser = argparse.ArgumentParser(description="Encode-Decode job")

    parser.add_argument(
        '-f', '--checkpoint_file', 
        type=str, 
        required=True,  # this should be true
        help='The checkpoint (ckpt) file for the model'
    )

    parser.add_argument(
        '-d', '--data_path', 
        type=str, 
        required=True,  # this should be true
        help='The path with the data'
    )

    parser.add_argument(
        '-i', '--num_iters', 
        type=int, 
        required=True,  # this should be true
        help='Number of decoding-encoding iterations'
    )

    parser.add_argument(
        '-s', '--grid_scale_factor', 
        type=float, 
        default=1.0,  # this should be true
        help='Scale factor to expand the grid by'
    )

    parser.add_argument(
        '-r', '--resolution', 
        type=int, 
        default=25,  # this should be true
        help='Resolution as number of points generate in linspace'
    )

    parser.add_argument(
        '-p', '--pdbs',
        type=str,
        nargs='+',  # Accepts one or more files
        help='Specify one or more PDBS'
    )

    parser.add_argument(
        '-o', '--output_dir',
        type=str,
        required=True,  # this should be true
        help='Specify the path to save any output'
    )

    parser.add_argument(
        '-a', '--autoencoder', 
        type=str, 
        required=True,
        help='The autoencoder type'
    )

    args = parser.parse_args()
    num_iters = args.num_iters
    scale_factor = args.grid_scale_factor
    res = args.resolution
    checkpoint_file = args.checkpoint_file
    data_path = args.data_path if args.data_path[-1] != '/' else args.data_path[:-1]
    datafiles = [f'{data_path}/{pdb}' for pdb in args.pdbs]
    output_dir = args.output_dir if args.output_dir[-1] != '/' else args.output_dir[:-1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autoencoder_of_choice = AUTOENCODER_SELLECTION[args.autoencoder]


    assert res > 1, "Resolution must be greater than 1 otherwise the code will fail"

    log_params(
        path=output_dir,
        experiment=sys.argv[0],
        output_dir=output_dir,
        datafiles=datafiles,
        autoencoder_of_choice=str(autoencoder_of_choice),
        checkpoint_file=checkpoint_file,
        resolution=res,
        scale_factor=scale_factor,
        num_iters=num_iters,
        python_version=sys.version,
        torch_version=torch.__version__,
        device=str(device),
    )


    torch.manual_seed(2025)
    print(f"{device=}")

    batch_size=8

    checkpoint = torch.load(checkpoint_file, map_location=torch.device('cpu'), weights_only=False)
    model = autoencoder_of_choice(**checkpoint['network_kwargs'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(device)

    data = get_data(datafiles).dataset
    data = data.to(device)
    num_atoms = data.size(1)

    initial_z = batched_encode(model=model, dataset=data, batch_size=batch_size, verbose=True)

    min_x = torch.min(initial_z.squeeze()[:, 0]) - (torch.min(initial_z.squeeze()[:, 0]) * scale_factor)
    min_y = torch.min(initial_z.squeeze()[:, 1]) - (torch.min(initial_z.squeeze()[:, 1]) * scale_factor)
    max_x = torch.max(initial_z.squeeze()[:, 0]) + (torch.max(initial_z.squeeze()[:, 0]) * scale_factor)
    max_y = torch.max(initial_z.squeeze()[:, 1]) + (torch.max(initial_z.squeeze()[:, 1]) * scale_factor)


    x_lin = torch.linspace(min_x, max_x, res, device=device)
    y_lin = torch.linspace(min_y, max_y, res, device=device)
    X, Y = torch.meshgrid(x_lin, y_lin, indexing='xy')

    startings = torch.stack([X.flatten(), Y.flatten()], dim=1)
    endings = decode_encode(model=model, z=startings[:, :, None], num_iters=num_iters, num_atoms=num_atoms, batch_size=batch_size, verbose=True)
    
    now = datetime.datetime.now()
    timestamp_format = "%Y%m%d_%H%M%S"
    timestamp = now.strftime(timestamp_format)

    torch.save(endings['encodings'][-1, :, :], f"{output_dir}/{timestamp}_encodings.pt")
    print(f"Saved encodings in {output_dir}/{timestamp}_encodings.pt")

    plot_drifting(z=endings['encodings'], num_iters=num_iters, output_dir=output_dir, res=res, min_x=None, max_x=max_x, min_y=min_y, max_y=max_y, gif=True)

    print("Script complete. Exiting.")
    return 0

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    duration_seconds = end_time - start_time
    minutes = math.floor(duration_seconds / 60) 
    remaining_seconds = duration_seconds % 60
    print(f"Python script duration: **{minutes} minutes and {remaining_seconds:.2f} seconds**")