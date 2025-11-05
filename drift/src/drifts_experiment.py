from utils import get_data, decode_encode, batched_encode, plot_drifting
import torch
from molearn.models.foldingnet import AutoEncoder
import argparse
import sys

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

    args = parser.parse_args()
    num_iters = args.num_iters
    scale_factor = args.grid_scale_factor
    res = args.resolution
    checkpoint_file = args.checkpoint_file
    data_path = args.data_path if args.data_path[-1] != '/' else args.data_path[:-1]
    datafiles = [f'{data_path}/{pdb}' for pdb in args.pdbs]
    output_dir = args.output_dir

    assert res > 1, "Resolution must be greater than 1 otherwise the code will fail"

    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")

    torch.manual_seed(2025)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{device=}")

    batch_size=8

    checkpoint = torch.load(checkpoint_file, map_location= torch.device('cpu'), weights_only=False)
    model = AutoEncoder(**checkpoint['network_kwargs'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(device)

    data = get_data(datafiles).dataset
    data = data.to(device)
    num_atoms = data.size(1)

    rand_indices = torch.randperm(data.shape[0], device=device)
    rand_dataset = data[rand_indices]

    initial_z = batched_encode(model=model, dataset=rand_dataset[0:50], batch_size=batch_size, verbose=True)

    min_x = torch.min(initial_z.squeeze()[:, 0]) * scale_factor
    min_y = torch.min(initial_z.squeeze()[:, 1]) * scale_factor
    max_x = torch.max(initial_z.squeeze()[:, 0]) * scale_factor
    max_y = torch.max(initial_z.squeeze()[:, 1]) * scale_factor


    x_lin = torch.linspace(min_x, max_x, res, device=device)
    y_lin = torch.linspace(min_y, max_y, res, device=device)
    X, Y = torch.meshgrid(x_lin, y_lin, indexing='xy')

    print(X.device)

    startings = torch.stack([X.flatten(), Y.flatten()], dim=1)
    endings = decode_encode(model=model, z=startings[:, :, None], num_iters=num_iters, num_atoms=num_atoms, batch_size=batch_size, verbose=True)

    plot_drifting(z=endings['encodings'], num_iters=num_iters, output_dir=output_dir)

    print("Script complete. Exiting.")
    return 0

if __name__ == "__main__":
    main()