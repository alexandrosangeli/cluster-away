import math
import os
import sys
import time
import torch
from molearn.analysis import MolearnAnalysis

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(current_dir, '..', '..') 
sys.path.append(root_dir)
from generic_utils.utils import AUTOENCODER_SELLECTION, get_data
from generic_utils.cli_utils import parse_all_args



import matplotlib.pyplot as plt


def main(args):
    device = args['device']
    training_datafiles = args['datafiles']
    autoencoder_of_choice = args['autoencoder_of_choice']
    timestamp = args['timestamp']
    request_gpu = args['request_gpu']
    description = args['description']
    verbose = args['verbose']
    output_dir = args['output_dir']
    

    # Experiment specific
    checkpoint_file = args['checkpoint_file']

    checkpoint = torch.load(checkpoint_file, map_location=torch.device('cpu'), weights_only=False)
    model = autoencoder_of_choice(**checkpoint['network_kwargs'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(device)

    data = get_data(training_datafiles)
    MA = MolearnAnalysis()
    MA.set_network(model)
    MA.set_dataset('training', data)

    # Starting analysis ops
    err = MA.get_error('training')

    print(f"{err.shape=}, {err.mean()=}")
    n_samples = 10

    grid_bounds = (14., 22., 0., -10.) # these just seemed to cover the DOPE space well
    MA.setup_grid(n_samples, bounds=grid_bounds)

    dope_grid_err, xs, ys = MA.scan_dope()
    print(f"{dope_grid_err.shape=}")
    plot_matrix(dope_grid_err, xs, ys, "DOPE", output_dir)

    rama_grid_err, xs, ys = MA.scan_ramachandran()
    print(f"{rama_grid_err.shape=}")
    plot_matrix(rama_grid_err.reshape(n_samples, n_samples), xs, ys, "Ramachandran", output_dir)


def plot_matrix(matrix, xs, ys, metric, output_dir, label_axis=False):
    fig, ax = plt.subplots()
    c = ax.pcolormesh(
        matrix, 
        cmap='viridis',
        shading='auto'
    )
    fig.colorbar(c, ax=ax, label=f'{metric} value')
    ax.set_title("title")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    N = len(xs)

    ax.set_xticks(torch.arange(N) + 0.5)
    ax.set_yticks(torch.arange(N) + 0.5)

    formatted_xs = [f"{x:.1f}" for x in xs]
    formatted_ys = [f"{y:.1f}" for y in ys]

    if label_axis: # false by default
        ax.set_xticklabels(formatted_xs)
        ax.set_yticklabels(formatted_ys)
        plt.setp(ax.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor")

    filename = f"{output_dir}/{metric}_plot.png"
    plt.savefig(filename)
    plt.close(fig)
    print(f"Saved {metric} plot in {filename}")
    

if __name__ == "__main__":
    start_time = time.time()
    args = parse_all_args(description="Analysis arg parser", experiment=sys.argv[0])
    main(args)
    end_time = time.time()
    duration_seconds = end_time - start_time
    minutes = math.floor(duration_seconds / 60) 
    remaining_seconds = duration_seconds % 60
    print(f"Python script duration: **{minutes} minutes and {remaining_seconds:.2f} seconds**")