#!/bin/bash

#  https://gist.github.com/sgraaf/5b0caa3a320f28c27c12b5efeb35aa4c

# Made to run on Metacentrum on a 2-GPU node.
export NUM_NODES=1
export NUM_GPUS_PER_NODE=2
export NODE_RANK=0
export WORLD_SIZE=$(($NUM_NODES * $NUM_GPUS_PER_NODE))

# launch your script w/ `torch.distributed.launch`
python -m torch.distributed.launch \
    --nproc_per_node=$NUM_GPUS_PER_NODE \
    --nnodes=$NUM_NODES \
    --node_rank $NODE_RANK \
    geopose/train.py --ddp --meta

