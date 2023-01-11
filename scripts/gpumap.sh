#!/bin/bash

gpu_map=(4 5 2 3 6 7 0 1)
gpus_per_node=8

export ROCR_VISIBLE_DEVICES=${gpu_map[SLURM_LOCALID%gpus_per_node]}

exec $*
