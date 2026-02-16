from molearn.models.CNN_autoencoder import AutoEncoder as ConvolutionalAE
from molearn.models.foldingnet import AutoEncoder as FoldingNet
from molearn.trainers import OpenMM_Physics_Trainer
from molearn.data import PDBData
import datetime
import time
import os
import sys
import math
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(current_dir, '..', '..') 
sys.path.append(root_dir)
from generic_utils.utils import AUTOENCODER_SELLECTION, AUTOENCODER_DEFAULT_MANDATORY_ARGUMENTS, get_data
from generic_utils.cli_utils import parse_all_args


def main(args):
    # parser = argparse.ArgumentParser(description="Model training job")

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
    model_kwargs = AUTOENCODER_DEFAULT_MANDATORY_ARGUMENTS[args['autoencoder']]
    physics_weight = args['physics_weight']
    patience = args['patience']
    batch_size = args['batch_size']

    data = get_data(datafiles, fix_terminal=True)
    data.write_statistics(f"{output_dir}/data_statistics.json") # Save mean and std for analysis later
    
    trainer = OpenMM_Physics_Trainer(device=device, physics_inter_weight=physics_weight)
    trainer.set_data(data, 
                     batch_size=batch_size, 
                     validation_split=0.1, 
                     manual_seed=25,
                     save_indices=True,
                     indices_dir=f"{output_dir}/indices"
                     )
    trainer.prepare_physics(remove_NB=True)
    trainer.set_autoencoder(autoencoder_of_choice, **model_kwargs)
    trainer.prepare_optimiser()

    trainer.run(
        epochs=10,
        log_filename="log.dat",
        log_folder=f"{output_dir}/logs",
        checkpoint_folder=f"{output_dir}/checkpoints",
        verbose=True,
    )

    physics_inter_weight = trainer.get_scale(
        ref_loss=trainer.results_epoch["mse_loss"],
        tar_loss=trainer.results_epoch["inter_physics_loss"],
        scale_scale=physics_weight
        )

    trainer.update_hyperparameters(physics_inter_weight=hysics_inter_weight)

    trainer.run_until_converge(
        patience=patience,
        log_filename="log.dat",
        log_folder=f"{output_dir}/logs",
        checkpoint_folder=f"{output_dir}/checkpoints",
        verbose=True,
    )

    print(fit_results)
    print("Script complete. Exiting.")
    return 0


if __name__ == "__main__":
    start_time = time.time()
    args = parse_all_args(description="Model training experiment arg parser", experiment=sys.argv[0])
    main(args)
    end_time = time.time()
    duration_seconds = end_time - start_time
    minutes = math.floor(duration_seconds / 60) 
    remaining_seconds = duration_seconds % 60
    print(f"Python script duration: **{minutes} minutes and {remaining_seconds:.2f} seconds**")