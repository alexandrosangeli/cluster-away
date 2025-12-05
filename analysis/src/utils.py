import torch
import matplotlib.pyplot as plt


def plot_violin_dope(MA, output_dir):
    dope_err = MA.get_dope('training')
    data_dope = dope_err["dataset_dope"]
    decoded_dope = dope_err["decoded_dope"]
    
    _plot_violin_internal(
        data_list=[data_dope, decoded_dope],
        y_label="DOPE / a.u.",
        xtick_labels=["Dataset DOPE", "Decoded DOPE"],
        colours=["orange", "cyan"],
        output_dir=output_dir,
        filename_prefix="dope"
    )

def plot_violin_rmsd(MA, output_dir):
    rmsd_err = MA.get_error('training')
    print(f"{rmsd_err.shape=}")
    
    _plot_violin_internal(
        data_list=[rmsd_err],
        y_label="RMSD / $\AA$",
        xtick_labels=["RMSD Error"],
        colours=["red"],
        output_dir=output_dir,
        filename_prefix="rmsd"
    )

def plot_dope_grid(MA, output_dir):
    dope_grid_err, xs, ys = MA.scan_dope()
    _plot_matrix_internal(dope_grid_err, xs, ys, "DOPE", output_dir, MA.get_encoded("training"))


def plot_rama_grid(MA, output_dir):
    rama_grid_err, xs, ys = MA.scan_ramachandran()
    _plot_matrix_internal(rama_grid_err, xs, ys, "Ramachandran", output_dir, MA.get_encoded("training"))

def _plot_violin_internal(data_list, y_label, xtick_labels, colours, output_dir, filename_prefix):
    fig, ax = plt.subplots(figsize=(6, 5))
    
    violins = ax.violinplot(
        data_list,
        showmeans=True,
        showextrema=True,
        widths=0.5
    )

    for i, violin in enumerate(violins['bodies']):
        color = colours[i % len(colours)]
        violin.set_color(color)
        violin.set_alpha(1)

    for partname in ('cbars', 'cmins', 'cmaxes', 'cmeans'):
        vp = violins[partname]
        vp.set_edgecolor("k")
        vp.set_linewidth(1)

    ax.set_ylabel(y_label)
    ax.set_xticks(torch.arange(1, len(data_list) + 1))
    ax.set_xticklabels(xtick_labels)
    ax.set_title(f"Distribution of {filename_prefix.replace('_', ' ').title()}")
    
    filename = f"{output_dir}/{filename_prefix}_violin_plot.png"
    plt.savefig(filename)
    plt.close(fig)
    print(f"Saved {filename}")


def _plot_matrix_internal(matrix, xs, ys, metric, output_dir, dataset_latent=None, label_axis=False):
    fig, ax = plt.subplots()
    c = ax.pcolormesh(
        xs,
        ys,
        matrix, 
        cmap='viridis',
        shading='auto'
    )
    if dataset_latent is not None:
        latent_x = dataset_latent[:, 0].cpu().numpy()
        latent_y = dataset_latent[:, 1].cpu().numpy()
        ax.scatter(
            latent_x,
            latent_y,
            marker='x',        
            color='red',       
            s=10,
            label='Latent Data Points',
            zorder=5
        )
    fig.colorbar(c, ax=ax, label=f'{metric} value')
    ax.set_title(f"{metric} Grid Scan")
    ax.set_xlabel("z_0")
    ax.set_ylabel("z_1")
    filename = f"{output_dir}/{metric}_grid_plot.png"
    plt.savefig(filename)
    plt.close(fig)
    print(f"Saved {filename}")


#####
# def plot_violin_dope(MA, output_dir):
#     dope_err = MA.get_dope('training')
#     data_dope = dope_err["dataset_dope"]
#     decoded_dope = dope_err["decoded_dope"]
#     fig = plt.figure()
#     ax = fig.add_subplot(1, 1, 1)
#     violins = ax.violinplot([data_dope, decoded_dope],
#                             showmeans=True,
#                             showextrema = True)

#     ax.set_ylabel("DOPE / a.u.")
#     ax.set_xticks([1, 2])
#     ax.set_xticklabels(["dataset_dope set", "decoded_dope decoded"])

#     colours=["orange", "cyan"]
#     for i, violin in enumerate(violins['bodies']):
#         violin.set_color(colours[i])
#         violin.set_alpha(1)

#     for partname in ('cbars','cmins','cmaxes','cmeans'):
#         vp = violins[partname]
#         vp.set_edgecolor("k")
#         vp.set_linewidth(1)

#     filename = f"{output_dir}/dope_violin_plot.png"
#     plt.savefig(filename)
#     plt.close(fig)
#     print(f"Saved {filename}")


# def plot_violin_rmsd(MA, output_dir):
#     rmsd_err = MA.get_error('training')
#     print(f"{rmsd_err.shape=}")
#     fig = plt.figure()
#     ax = fig.add_subplot(1, 1, 1)
#     violins = ax.violinplot([rmsd_err],
#                             showmeans=True,
#                             showextrema=True)

#     ax.set_ylabel("RMSD / $\AA$")
#     ax.set_xticks([1, 2])

#     colours=["red", "orange", "blue", "cyan"]
#     for i, violin in enumerate(violins['bodies']):
#         violin.set_color(colours[i])
#         violin.set_alpha(1)

#     for partname in ('cbars','cmins','cmaxes','cmeans'):
#         vp = violins[partname]
#         vp.set_edgecolor("k")
#         vp.set_linewidth(1)

#     filename = f"{output_dir}/rmsd_violin_plot.png"
#     plt.savefig(filename)
#     plt.close(fig)
#     print(f"Saved {filename}")


# def plot_dope_grid(MA, output_dir):
#     dope_grid_err, xs, ys = MA.scan_dope()
#     plot_matrix(dope_grid_err, xs, ys, "DOPE", output_dir, MA.get_encoded("training"))


# def plot_rama_grid(MA, output_dir):
#     rama_grid_err, xs, ys = MA.scan_ramachandran()
#     plot_matrix(rama_grid_err, xs, ys, "Ramachandran", output_dir, MA.get_encoded("training"))


# def plot_matrix(matrix, xs, ys, metric, output_dir, dataset_latent=None, label_axis=False):
#     fig, ax = plt.subplots()
#     c = ax.pcolormesh(
#         matrix, 
#         cmap='viridis',
#         shading='auto'
#     )

#     if dataset_latent is not None:
#         latent_x = dataset_latent[:, 0].cpu().numpy() + 0.5 # Add 0.5 to center them in the cells
#         latent_y = dataset_latent[:, 1].cpu().numpy() + 0.5
        
#         ax.scatter(
#             latent_x,
#             latent_y,
#             marker='x',         # Use a circular marker
#             color='red',        # Use a distinct color
#             s=10,               # Set marker size
#             edgecolor='white',  # Add a white border for visibility
#             label='Coordinates' # Add a label for the legend
#         )

#     fig.colorbar(c, ax=ax, label=f'{metric} value')
#     ax.set_title("title")
#     ax.set_xlabel("x")
#     ax.set_ylabel("y")
#     N = len(xs)

#     ax.set_xticks(torch.arange(N) + 0.5)
#     ax.set_yticks(torch.arange(N) + 0.5)

#     formatted_xs = [f"{x:.1f}" for x in xs]
#     formatted_ys = [f"{y:.1f}" for y in ys]

#     if label_axis: # false by default
#         ax.set_xticklabels(formatted_xs)
#         ax.set_yticklabels(formatted_ys)
#         plt.setp(ax.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor")

#     filename = f"{output_dir}/{metric}_grid_plot.png"
#     plt.savefig(filename)
#     plt.close(fig)
#     print(f"Saved {filename}")    