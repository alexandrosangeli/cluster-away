#!/bin/bash

#SBATCH -o /home/%u/slogs/sl_%A.out
#SBATCH -e /home/%u/slogs/sl_%A.out
#SBATCH -N 1	  # nodes requested
#SBATCH -n 1	  # tasks requested
#SBATCH --gres=gpu:1  # use 1 GPU
#SBATCH --mem=14000  # memory in Mb
#SBATCH --partition=PGR-Standard
#SBATCH -t 2:00:00  # time requested in hour:minute:seconds
#SBATCH --cpus-per-task=4

run_experiment() {
    python3 src/train.py \
        --output_dir=${OUTPUT_DIR} \
        --data_path=${DATA_SCRATCH}/proteins \
        --pdbs MurD_closed.pdb MurD_open.pdb \
        --autoencoder=fold_net \
        --timestamp=${TIMESTAMP} \
        --request_gpu=1 \
        --verbose=0
}

source ../main_script/main_script.sh