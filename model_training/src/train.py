from molearn.models.CNN_autoencoder import AutoEncoder as ConvolutionalAE
from molearn.models.foldingnet import AutoEncoder as FoldingNet
from molearn.trainers import OpenMM_Physics_Trainer
from molearn.data import PDBData
import datetime
import time
import math
import torch
import argparse


def log_params(**params):
    for p, v in params.items():
        print(f"{p}={v}")

def main():
    parser = argparse.ArgumentParser(description="Model training job")

    parser.add_argument(
        '-o', '--output_dir',
        type=str,
        required=True
    )

    parser.add_argument(
        '-d', '--data_path', 
        type=str, 
        required=True,
        help='The path with the data'
    )

    parser.add_argument(
        '-p', '--pdbs',
        type=str,
        nargs='+',  # Accepts one or more files
        help='Specify one or more PDBS'
    )


    args = parser.parse_args()
    output_dir = args.output_dir if args.output_dir[-1] != '/' else args.output_dir[:-1]
    data_path = args.data_path if args.data_path[-1] != '/' else args.data_path[:-1]
    datafiles = [f'{data_path}/{pdb}' for pdb in args.pdbs]

    log_params(
        output_dir=output_dir,
        datafiles=datafiles
    )

    data = PDBData()
    data.import_pdb(datafiles)
    data.fix_terminal()
    data.atomselect(atoms=["N", "CA", "CB", "C", "O"])
    dataset = data.prepare_dataset()
    data.write_statistics(f"{output_dir}/data_statistics.json") # Save mean and std for analysis later


    ##### Prepare Trainer #####
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = OpenMM_Physics_Trainer(device=device)

    trainer.set_data(data, 
                     batch_size=16, 
                     validation_split=0.1, 
                     manual_seed=25,
                     save_indices=False     # If True, the training/validation split indices will be saved to disk
                     )
    trainer.prepare_physics(remove_NB=True)
    trainer.set_autoencoder(AutoEncoder)
    trainer.prepare_optimiser()

    ##### Training Loop #####
    # Keep training until loss does not improve for 16 consecutive epochs

    now = datetime.datetime.now()
    timestamp_format = "%Y%m%d_%H%M%S"
    timestamp = now.strftime(timestamp_format)

    fit_results = trainer.run_until_converge(
        patience=16,
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