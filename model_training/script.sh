#!/bin/bash

bash ../slurm.sh "python3 train.py \
    --output_dir=out \
    --data_path=../drift/data/proteins/ \
    --pdbs MurD_closed.pdb MurD_open.pdb" "$@" 
