#!/bin/bash

# runs training using ibm's ddlrun with arguments

ddlrun --tcp --H cap --accelerators 4 python ./cli-training-ibm-ddlrun.py -p /gpfs/CompMicro/projects/nVidia_POC/dynamorph/microglia -o /gpfs/CompMicro/projects/nVidia_POC/dynamorph/microglia/JUNE/nvidia-poc-4GPU -c 1


# ddlrun --accelerators 4 python ./cli-training-ibm-ddlrun.py -p /gpfs/CompMicro/projects/nVidia_POC/dynamorph/microglia -o /gpfs/CompMicro/projects/nVidia_POC/dynamorph/microglia/JUNE/nvidia-poc-4GPU -c 1
