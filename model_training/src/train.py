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
from generic_utils.utils import log_params, AUTOENCODER_SELLECTION, AUTOENCODER_DEFAULT_MANDATORY_ARGUMENTS


def main():
    parser = argparse.ArgumentParser(description="Model training job")

    parser.add_argument(
        '--output_dir',
        type=str,
        required=True
    )

    parser.add_argument(
        '--data_path', 
        type=str, 
        required=True,
        help='The path with the data'
    )

    parser.add_argument(
        '--pdbs',
        type=str,
        nargs='+',  # Accepts one or more files
        help='Specify one or more PDBS'
    )

    parser.add_argument(
        '--autoencoder', 
        type=str, 
        required=True,
        help='The autoencoder type'
    )

    parser.add_argument(
        '--timestamp', 
        type=str, 
        required=True,
        help='The current timestamp'
    )

    parser.add_argument(
        '--description', 
        type=str, 
        default="",
        help='Optional description of the current experiment'
    )

    parser.add_argument(
        '--request_gpu', 
        type=int, 
        required=True,
        help='Flag 0/1 whether GPU was requested'
    )

    args = parser.parse_args()
    output_dir = args.output_dir if args.output_dir[-1] != '/' else args.output_dir[:-1]
    data_path = args.data_path if args.data_path[-1] != '/' else args.data_path[:-1]
    datafiles = [f'{data_path}/{pdb}' for pdb in args.pdbs]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autoencoder_of_choice = AUTOENCODER_SELLECTION[args.autoencoder]
    model_kwargs = AUTOENCODER_DEFAULT_MANDATORY_ARGUMENTS[args.autoencoder]
    patience = 16
    timestamp = args.timestamp
    request_gpu = args.request_gpu


    log_params(
        path=output_dir,
        experiment=sys.argv[0],
        output_dir=output_dir,
        datafiles=datafiles,
        autoencoder_of_choice=str(autoencoder_of_choice),
        model_kwargs=model_kwargs,
        patience=patience,
        python_version=sys.version,
        torch_version=torch.__version__,
        device=str(device),
        request_gpu=request_gpu,
        description=args.description,
    )

    assert request_gpu in [0, 1], "--request_gpu should take the value of 0 or 1"
    assert (not request_gpu) or (torch.cuda.is_available() == request_gpu), "GPU was requested but is not available"


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
    main()
    end_time = time.time()
    duration_seconds = end_time - start_time
    minutes = math.floor(duration_seconds / 60) 
    remaining_seconds = duration_seconds % 60
    print(f"Python script duration: **{minutes} minutes and {remaining_seconds:.2f} seconds**")