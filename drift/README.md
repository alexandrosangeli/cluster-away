# Generate latent space trajectories
`src/drift_experiment.py` generates a plot (saved in `output_dir`) of the trajectories of latent points in the latent space in `num_iters` iterations.

## Usage
To run the script run `script.sh` using `bash script.sh`. You can pass arguments to the python script `drift_experiment` from `script.sh` that set the `num_iters`, the resolution of the grid, the size of the grid (using `grid_scale_factor`), and more. 

### CUDA
To run the script using a GPU use the `-g` flag: `bash script.sh -g`.