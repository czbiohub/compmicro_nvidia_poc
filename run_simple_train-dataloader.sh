#!/bin/bash

# runs training with arguments

python cli-simple_train-dataloader.py \
-p /gpfs/CompMicro/projects/nVidia_POC/dynamorph/microglia \
-o /gpfs/CompMicro/projects/nVidia_POC/dynamorph/microglia/JUNE/nvidia-poc-1GPU-loader \
-c 1 \
-d "cuda:3"
