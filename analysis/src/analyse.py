import math
import os
import sys
import time
import torch
from molearn.analysis import MolearnAnalysis
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(current_dir, '..', '..') 
sys.path.append(root_dir)
from generic_utils.utils import AUTOENCODER_SELLECTION, get_data
from generic_utils.cli_utils import parse_all_args


def main(args):
    device = args['device']
    training_datafiles = args['datafiles']
    autoencoder_of_choice = args['autoencoder_of_choice']
    timestamp = args['timestamp']
    request_gpu = args['request_gpu']
    description = args['description']
    verbose = args['verbose']
    output_dir = args['output_dir']
    checkpoint_file = args['checkpoint_file']
    # Experiment specific
    n_samples = args['resolution']
    grid_bounds = (14, 22, 0, -10) # these just seemed to cover the DOPE space well
    grid_bounds=None

    checkpoint = torch.load(checkpoint_file, map_location=torch.device('cpu'), weights_only=False)
    model = autoencoder_of_choice(**checkpoint['network_kwargs'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(device)

    data = get_data(training_datafiles)
    data.dataset = data.dataset[:8]
    MA = MolearnAnalysis(batch_size=1, processes=1)
    MA.set_network(model)
    MA.set_dataset('training', data)
    MA.setup_grid(n_samples, bounds=grid_bounds)

    # plot_violin_rmsd(MA=MA, output_dir=output_dir)
    # plot_violin_dope(MA=MA, output_dir=output_dir)

    plot_dope_grid(MA=MA, output_dir=output_dir)
    # plot_rama_grid(MA=MA, output_dir=output_dir)

    MA.get_decoded

    return 0


def plot_violin_dope(MA, output_dir):
    dope_err = MA.get_dope('training')
    data_dope = dope_err["dataset_dope"]
    decoded_dope = dope_err["decoded_dope"]
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    violins = ax.violinplot([data_dope, decoded_dope],
                            showmeans=True,
                            showextrema = True)

    ax.set_ylabel("DOPE / a.u.")
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["dataset_dope set", "decoded_dope decoded"])


    # colours=["red", "orange", "blue", "cyan"]
    # for i, violin in enumerate(violins['bodies']):
    #     violin.set_color(colours[i])
    #     violin.set_alpha(1)

    for partname in ('cbars','cmins','cmaxes','cmeans'):
        vp = violins[partname]
        vp.set_edgecolor("k")
        vp.set_linewidth(1)
    plt.show()


def plot_violin_rmsd(MA, output_dir):
    rmsd_err = MA.get_error('training')
    print(f"{rmsd_err.shape=}")
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    violins = ax.violinplot([rmsd_err],
                            showmeans=True,
                            showextrema=True)

    ax.set_ylabel("RMSD / $\AA$")
    ax.set_xticks([1, 2])

    colours=["red", "orange", "blue", "cyan"]
    for i, violin in enumerate(violins['bodies']):
        violin.set_color(colours[i])
        violin.set_alpha(1)

    for partname in ('cbars','cmins','cmaxes','cmeans'):
        vp = violins[partname]
        vp.set_edgecolor("k")
        vp.set_linewidth(1)
    plt.show()


def plot_dope_grid(MA, output_dir):
    dope_grid_err, xs, ys = MA.scan_dope()
    plot_matrix(dope_grid_err, xs, ys, "DOPE", output_dir, MA.get_encoded("training"))


def plot_rama_grid(MA, output_dir):
    rama_grid_err, xs, ys = MA.scan_ramachandran()
    plot_matrix(rama_grid_err, xs, ys, "Ramachandran", output_dir, MA.get_encoded("training"))


def plot_matrix(matrix, xs, ys, metric, output_dir, dataset_latent=None, label_axis=False):
    fig, ax = plt.subplots()
    c = ax.pcolormesh(
        matrix, 
        cmap='viridis',
        shading='auto'
    )

    if dataset_latent is not None:
        latent_x = dataset_latent[:, 0].cpu().numpy() + 0.5 # Add 0.5 to center them in the cells
        latent_y = dataset_latent[:, 1].cpu().numpy() + 0.5
        
        ax.scatter(
            latent_x,
            latent_y,
            marker='x',         # Use a circular marker
            color='red',        # Use a distinct color
            s=10,               # Set marker size
            edgecolor='white',  # Add a white border for visibility
            label='Coordinates' # Add a label for the legend
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