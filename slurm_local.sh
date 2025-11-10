#!/bin/bash

set -e # fail fast


# Check for the -g flag
if [[ "$*" == *"-g"* ]]; then
    CONDA_ENV_NAME="molearn-gpu"
else
    CONDA_ENV_NAME="molearn"
fi

export MOLEARN_PATH=/home/${USER}/repos/molearn
source /home/${USER}/miniconda3/bin/activate ${CONDA_ENV_NAME}

echo "Activated ${CONDA_ENV_NAME}"

if python3 -c "import molearn" 2>/dev/null; then
    echo "Molearn is already installed."
else
    echo "Molearn not found. Installing from source..."
    python3 -m pip install "$MOLEARN_PATH"
    echo "Molearn nstallation complete."
fi

dt=$(date '+%d_%m_%y_%H_%M');
echo "I am job ${SLURM_JOB_ID}"
echo "I'm running on ${SLURM_JOB_NODELIST}"
echo "Job started at ${dt}"

export DATA_SCRATCH=data/

export OUTPUT_DIR=experiments/
mkdir -p ${OUTPUT_DIR}

echo "Running Python script..."
eval "$1"


echo "Job is done"
exit 0