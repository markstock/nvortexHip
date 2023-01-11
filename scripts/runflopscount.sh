#!/bin/sh -l

TASKS="-n1 -N1 -pbardpeak --exclusive"
CBIND="--cpus-per-task=8 --threads-per-core=1"
#CBIND="-c 8 --cpu-bind=map_cpu:1 --mem-bind=local --hint=nomultithread"
#GBIND="--gpus-per-node=8 --gpus-per-task=1 --gpu-bind=closest"
PROF="rocprof --verbose --timestamp on --stats -m ./metrics.xml -i ./metrics.txt"
EXE="./ngHip06.bin -n=400000"
srun $TASKS $CBIND $GBIND $PROF $EXE

