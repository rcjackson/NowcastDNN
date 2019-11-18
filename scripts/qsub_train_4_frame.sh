#!/bin/bash
#COBALT --time=12:00:00
#COBALT -n 1
echo "Starting Jupyter notebook server"
NODES=`cat $COBALT_NODEFILE | wc -l`
PROCS=$((NODES * 12))
source activate pyart_ml
python train_radar_old_epoch.py 100 4
