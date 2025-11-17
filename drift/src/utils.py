import torch
from molearn.data import PDBData
import matplotlib.pyplot as plt
import datetime
from tqdm import tqdm
import os


def get_data(datafiles, atoms_select=True):
    print("Loading data...")
    data = PDBData()
    data.import_pdb(datafiles)
    if atoms_select:
        data.atomselect(atoms=['CA', 'C', 'CB', 'N', 'O'])
    data.prepare_dataset()
    print("Data loaded.")
    return data


def plot_z(z):
    """ z is a torch tensor (num_z, 2)"""
    # data = torch.stack(z).detach().numpy()
    data = z.detach().numpy()
    plt.figure(figsize=(8, 6))
    plt.scatter(data[:, 0], data[:, 1], marker='o')
    plt.title('')
    plt.xlabel('z_1')
    plt.ylabel('z_2')
    plt.grid(False)
    plt.show()

def calculate_drift_coords(start, end, norm=False):
    start = start.detach()
    end = end.detach()

    X = start[:, 0]
    Y = start[:, 1]
    U = end[:, 0] - X
    V = end[:, 1] - Y
    return {"start" : (X, Y), "end" : (U, V)}
    

def plot_drifting(z, num_iters, output_dir, res, min_x=None, max_x=None, min_y=None, max_y=None, gif=True):
    z = z.cpu()
    alpha = (-1/30000) * (res**2) + 0.1 # Dynamic alpha based on the resolution
    fig, ax = plt.subplots(figsize=(8, 8)) # Use a smaller figure size for GIF frames
    
    now = datetime.datetime.now()
    timestamp_format = "%Y%m%d_%H%M%S"
    timestamp = now.strftime(timestamp_format)

    if None not in [min_x, max_x, min_y, max_y]:
        x_min, x_max, y_min, y_max = min_x.cpu(), max_x.cpu(), min_y.cpu(), max_y.cpu()
        padding_x = 0.1 * (x_max - x_min)
        padding_y = 0.1 * (y_max - y_min)
        ax.set_xlim(x_min - padding_x, x_max + padding_x)
        ax.set_ylim(y_min - padding_y, y_max + padding_y)

    # Iterate from the first step (i=0) up to the final step (i=num_iters - 1)
    for i in range(num_iters):
        X = z[i].squeeze()
        X_transformed = z[i+1].squeeze()

        if i == 0:
            ax.scatter(X[:, 0], X[:, 1], color='blue', marker='x', s=5, label='Start Points', zorder=3)

        for j in range(X.shape[0]):
            start_x, start_y = X[j]
            end_x, end_y = X_transformed[j]
            ax.plot([start_x, end_x], [start_y, end_y], 'k-', alpha=alpha, linewidth=1, zorder=1)

        if gif:
            curr_end_points = ax.scatter(X_transformed[:, 0], X_transformed[:, 1], color='red', marker='o', s=10, label='End Points', zorder=3)
            os.makedirs(f'{output_dir}/gif_{timestamp}', exist_ok=True)
            frame_filename = f'{output_dir}/gif_{timestamp}/frame_{i:04d}.png'
            ax.set_title(f'Trajectories (Iteration {i+1} / {num_iters})')
            ax.legend()
            plt.savefig(frame_filename, dpi=150)
            print(f"Saved GIF frame: {frame_filename}")
            curr_end_points.remove()
        
    ax.scatter(X_transformed[:, 0], X_transformed[:, 1], color='red', marker='o', s=10, label='End Points', zorder=3)

    
    ax.set_title(f'Trajectories (Complete)')
    ax.set_xlabel('z_1')
    ax.set_ylabel('z_2')
    ax.grid(True, linestyle='', alpha=0.5)
    ax.set_aspect('equal', adjustable='box') 

    final_filename = f'{output_dir}/{timestamp}_trajectories_plot.svg'
    print(f"Saving final plot in {final_filename}")
    plt.savefig(final_filename, format='svg')
    plt.close(fig)
    
    if gif:
        print(f"All {num_iters} GIF frames and the final SVG plot have been saved to {output_dir}.")

# def plot_drifting(z, num_iters, output_dir, res, min_x=None, max_x=None, min_y=None, max_y=None):
#     fig, ax = plt.subplots(figsize=(30, 30))
#     z = z.cpu()
#     alpha = (-1/30000) * (res**2) + 0.1 # Dynamic alpha based on the resolution
#     for i in range(num_iters):
#         X = z[i].squeeze()
#         X_transformed = z[i+1].squeeze()

#         X_np = X.numpy()
#         X_transformed_np = X_transformed.numpy()

#         if i == 0:
#             ax.scatter(X_np[:, 0], X_np[:, 1], color='blue', marker='x', s=5, label='Start Points', zorder=3)

#         for j in range(X.shape[0]):
#             # Get the start and end coordinates for the i-th point
#             start_x, start_y = X_np[j]
#             end_x, end_y = X_transformed_np[j]

#             # Plot the line segment from start to end
#             # 'k-' is a black solid line
#             ax.plot([start_x, end_x], [start_y, end_y], 'k-', alpha=alpha, linewidth=1, zorder=1)

#         if i == num_iters - 1:
#             ax.scatter(X_transformed_np[:, 0], X_transformed_np[:, 1], color='red', marker='o', s=10, label='End Points', zorder=3)

#     if None not in [min_x, max_x, min_y, max_y]:
#         min_x, max_x, min_y, max_y = min_x.cpu(), max_x.cpu(), min_y.cpu(), max_y.cpu()
#         padding_x = 0.1 * (x_max - x_min)
#         padding_y = 0.1 * (y_max - y_min)
#         ax.set_xlim(min_x - padding, max_x + padding)
#         ax.set_ylim(min_y - padding, max_y + padding)

#     now = datetime.datetime.now()
#     timestamp_format = "%Y%m%d_%H%M%S"
#     timestamp = now.strftime(timestamp_format)

#     ax.set_title(f'Trajectories')
#     ax.set_xlabel('z_1')
#     ax.set_ylabel('z_2')
#     ax.legend()
#     ax.grid(True, linestyle='', alpha=0.5)
#     ax.set_aspect('equal', adjustable='box') # Ensures accurate visual representation of distances

#     filename = f'{output_dir}/{timestamp}_trajectories_plot.svg'
#     print(f"Saving plot in {filename}")
#     plt.savefig(filename, format='svg')
#     plt.close()


def batched_encode(model, dataset, batch_size, z_dim=2, verbose=False):
    # Preallocate space for the results to go
    z = torch.empty(dataset.shape[0], z_dim, dtype=dataset.dtype, device = dataset.device)
    iterator = range(0, z.shape[0], batch_size)
    iterator = tqdm(iterator) if verbose else iterator
    with torch.no_grad():
        for i in iterator:
            z[i:i+batch_size] = model.encode(dataset[i:i+batch_size].float()).squeeze()
    return z

def batched_decode(model, latent_vector, n_atoms, batch_size, verbose=False):
    decoded = torch.empty(latent_vector.shape[0], 3, n_atoms, dtype=latent_vector.dtype, device = latent_vector.device)
    iterator = range(0, decoded.shape[0], batch_size)
    iterator = tqdm(iterator) if verbose else iterator
    with torch.no_grad():
        for i in iterator:
            decoded[i:i+batch_size] = model.decode(latent_vector[i:i+batch_size].float())[:, :n_atoms, :].permute(0, 2, 1)
    return decoded


def encode_decode(model, dataset, N, num_atoms, batch_size, verbose=False):
    """
    Encode-Decode data N times
    model: model with loaded weights
    data: torch tensor
    """
    
    z = []
    decodings = [dataset]
    for i in range(N):
        if verbose:
            print(f"Encode-Decode iteration: [{i+1}/{N}]")
        encoded = batched_encode(model, dataset, batch_size)
        z.append(encoded)
        decoded = batched_decode(model, encoded, num_atoms, batch_size).permute(0,2,1)
        decodings.append(decoded)
        dataset = decoded
    
    return {"encodings" : torch.stack(z), "decodings" : torch.stack(decodings)}


def decode_encode(model, z, num_iters, num_atoms, batch_size, verbose=False):
    """
    Encode-Decode data N times
    model: model with loaded weights
    data: torch tensor
    """
    
    zs = [z]
    decodings = []
    for i in range(num_iters):
        if verbose:
            print(f"Decode-Encode iteration: [{i+1}/{num_iters}]")
        decoded = batched_decode(model, z, num_atoms, batch_size).permute(0,2,1)
        decodings.append(decoded)
        encoded = batched_encode(model, decoded, batch_size)
        zs.append(encoded)
        z = encoded

    return {"encodings" : torch.stack(zs), "decodings" : torch.stack(decodings)}


