#!/bin/bash

#SBATCH -o /home/%u/slogs/sl_%A.out
#SBATCH -e /home/%u/slogs/sl_%A.out
#SBATCH -N 1	  # nodes requested
#SBATCH -n 1	  # tasks requested
#SBATCH --gres=gpu:1  # use 1 GPU
#SBATCH --mem=14000  # memory in Mb
#SBATCH --partition=PGR-Standard
#SBATCH -t 00:30:00  # time requested in hour:minute:seconds
#SBATCH --cpus-per-task=32

run_experiment() {
    python3 src/analyse.py \
        --checkpoint_file=${DATA_SCRATCH}/models/checkpoints/foldingnet/checkpoint_no_optimizer_state_dict_epoch167_loss0.003259085263643.ckpt \
        --data_path=${DATA_SCRATCH}/proteins \
        --pdbs MurD_closed.pdb MurD_open.pdb \
        --output_dir=${OUTPUT_DIR} \
        --autoencoder=fold_net \
        --resolution=40 \
        --grid_scale_factor=0.2 \
        --num_cores=${NUM_CORES} \
        --batch_size=16 \
        --timestamp=${TIMESTAMP} \
        --request_gpu=1 \
        --verbose=0 \
        --description="Analysing model" \
        --molearn_git="Branch:Commit ID --> ${MOLEARN_BRANCH}:${MOLEARN_COMMITID}" \
        --cluster_away_git="This Commit ID: ${THIS_COMMITID}"
}

source ../main_script/main_script.sh