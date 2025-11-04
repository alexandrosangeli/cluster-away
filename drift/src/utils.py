import torch
from molearn.data import PDBData
import matplotlib.pyplot as plt
import datetime
from tqdm import tqdm


def get_data(datafiles, atoms_select=True):
    print("Loading data...")
    data = PDBData()
    for f in datafiles:
        data.import_pdb(f)
    if atoms_select:
        data.atomselect(atoms=['CA', 'C', 'CB', 'N', 'O'])
    data.prepare_dataset()
    print("Done!")
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
    

def plot_drifting(z, num_iters):
    fig, ax = plt.subplots(figsize=(8, 8))
    print(f"{z.shape=}")
    for i in range(num_iters):
        X = z[i].squeeze()
        print(f"{X.shape=}")
        X_transformed = z[i+1].squeeze()

        X_np = X.numpy()
        X_transformed_np = X_transformed.numpy()

        # Iterate and Plot N Lines
        for j in range(X.shape[0]):
            # Get the start and end coordinates for the i-th point
            start_x, start_y = X_np[j]
            end_x, end_y = X_transformed_np[j]

            # Plot the line segment from start to end
            # 'k-' is a black solid line
            ax.plot([start_x, end_x], [start_y, end_y], 'k-', alpha=0.15, linewidth=1, zorder=1)

        if i == num_iters - 1:
            ax.scatter(X_transformed_np[:, 0], X_transformed_np[:, 1], color='red', marker='o', s=10, label='End Points', zorder=3)

    now = datetime.datetime.now()
    timestamp_format = "%Y%m%d_%H%M%S"
    timestamp = now.strftime(timestamp_format)

    ax.set_title(f'Trajectories')
    ax.set_xlabel('z_1')
    ax.set_ylabel('z_2')
    ax.legend()
    ax.grid(True, linestyle='', alpha=0.5)
    ax.set_aspect('equal', adjustable='box') # Ensures accurate visual representation of distances
    plt.savefig(f'{timestamp}_trajectories_plot.png')
    plt.close()


def batched_encode(model, dataset, batch_size, verbose=False):
    # Preallocate space for the results to go
    if verbose:
        print("Encoding...")
    z = torch.empty(dataset.shape[0], 2, 1, dtype=dataset.dtype, device = dataset.device)
    with torch.no_grad():
        for i in tqdm(range(0, z.shape[0], batch_size)):
            z[i:i+batch_size] = model.encode(dataset[i:i+batch_size].float())
    return z

def batched_decode(model, latent_vector, n_atoms, batch_size, verbose=False):
    if verbose:
        print("Decoding...")
    decoded = torch.empty(latent_vector.shape[0], 3, n_atoms, dtype=latent_vector.dtype, device = latent_vector.device)
    with torch.no_grad():
        for i in tqdm(range(0, decoded.shape[0], batch_size)):
            decoded[i:i+batch_size] = model.decode(latent_vector[i:i+batch_size].float())[:,:,:n_atoms]
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
        decoded = batched_decode(model, encoded, num_atoms, batch_size)
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
        decoded = batched_decode(model, z, num_atoms, batch_size)
        decodings.append(decoded)
        encoded = batched_encode(model, decoded, batch_size)
        zs.append(encoded)
        z = encoded

    return {"encodings" : torch.stack(zs), "decodings" : torch.stack(decodings)}
