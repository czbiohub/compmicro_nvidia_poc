#!/bin/bash

# runs training with arguments

python ./dl_training/dynamorph/vqvae/cli.py -p /gpfs/CompMicro/projects/nVidia_POC/dynamorph/microglia -o /gpfs/CompMicro/projects/nVidia_POC/dynamorph/microglia/JUNE/trained_models -c [1] -g "cuda:1"
