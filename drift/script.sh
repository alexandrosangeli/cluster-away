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

set -e # fail fast

# source /home/${USER}/miniconda3/bin/activate molearn
source /home/${USER}/anaconda3/bin/activate molearn


dt=$(date '+%d_%m_%y_%H_%M');
echo "I am job ${SLURM_JOB_ID}"
echo "I'm running on ${SLURM_JOB_NODELIST}"
echo "Job started at ${dt}"

# ====================
# RSYNC data from /home/ to /disk/scratch/
# ====================
export SCRATCH_HOME=/disk/scratch/${USER}
export DATA_HOME=${PWD}/data
export DATA_SCRATCH=${SCRATCH_HOME}/experiments/data
mkdir -p ${SCRATCH_HOME}/experiments/data
rsync --archive --update --compress --progress ${DATA_HOME}/ ${DATA_SCRATCH}


echo "Creating directory to save the plot"
export OUTPUT_DIR=${SCRATCH_HOME}/experiments/plots
mkdir -p ${OUTPUT_DIR}


python3 src/drifts_experiment.py \
    --checkpoint_file=${DATA_SCRATCH}/models/checkpoints/foldingnet/checkpoint_no_optimizer_state_dict_epoch167_loss0.003259085263643.ckpt \
    --data_path=${DATA_SCRATCH}/proteins \
    --num_iters=1 \
    --grid_scale_factor=1.0 \
    --resolution=25 \
    --pdbs MurD_closed.pdb MurD_open.pdb

OUTPUT_HOME=${PWD}/plots/
mkdir -p ${OUTPUT_HOME}
rsync --archive --update --compress --progress ${OUTPUT_DIR} ${OUTPUT_HOME}

rm -rf ${OUTPUT_DIR}

echo "Job is done"
exit 0