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
import argparse

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(current_dir, '..', '..') 
sys.path.append(root_dir)
from generic_utils.utils import AUTOENCODER_SELLECTION, AUTOENCODER_DEFAULT_MANDATORY_ARGUMENTS
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
    patience = 16

    data = PDBData()
    data.import_pdb(datafiles)
    data.fix_terminal()
    data.atomselect(atoms=["N", "CA", "CB", "C", "O"])
    data.prepare_dataset()
    data.write_statistics(f"{output_dir}/data_statistics.json") # Save mean and std for analysis later
    
    trainer = OpenMM_Physics_Trainer(device=device)
    trainer.set_data(data, 
                     batch_size=16, 
                     validation_split=0.1, 
                     manual_seed=25,
                     save_indices=False     # If True, the training/validation split indices will be saved to disk
                     )
    trainer.prepare_physics(remove_NB=True)
    trainer.set_autoencoder(autoencoder_of_choice, **model_kwargs)
    trainer.prepare_optimiser()

    fit_results = trainer.run_until_converge(
        patience=patience,
        log_filename="log.dat",
        log_folder=f"{output_dir}/checkpoints_{timestamp}",
        checkpoint_folder=f"{output_dir}/checkpoints_{timestamp}",
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