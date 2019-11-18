#!/bin/bash
#COBALT --time=5:00:00
#COBALT -n 1
echo "Starting Jupyter notebook server"
NODES=`cat $COBALT_NODEFILE | wc -l`
PROCS=$((NODES * 12))
cd ~
source activate pyart_ml
jupyter notebook --port 8080
