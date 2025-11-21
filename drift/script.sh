#!/bin/bash

#SBATCH -o /home/%u/slogs/sl_%A.out
#SBATCH -e /home/%u/slogs/sl_%A.out
#SBATCH -N 1	  # nodes requested
#SBATCH -n 1	  # tasks requested
#SBATCH --gres=gpu:1  # use 1 GPU
#SBATCH --mem=24000  # memory in Mb
#SBATCH --partition=PGR-Standard
#SBATCH -t 2:00:00  # time requested in hour:minute:seconds
#SBATCH --cpus-per-task=8

run_experiment() {
    python3 src/drifts_experiment.py \
        --checkpoint_file=${DATA_SCRATCH}/models/checkpoints/foldingnet/checkpoint_no_optimizer_state_dict_epoch167_loss0.003259085263643.ckpt \
        --data_path=${DATA_SCRATCH}/proteins \
        --num_iters=100 \
        --grid_scale_factor=0.5 \
        --resolution=50 \
        --pdbs MurD_closed.pdb MurD_open.pdb \
        --output_dir=${OUTPUT_DIR} \
        --autoencoder=fold_net \
        --timestamp=${TIMESTAMP} \
        --request_gpu=1 \
        --verbose=0 \
        --gif=0
}

source ../main_script/main_script.sh