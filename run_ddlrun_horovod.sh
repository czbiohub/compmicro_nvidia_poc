#!/bin/bash

# runs training using ibm's ddlrun with arguments
#
#ddlrun --H cap \
#--accelerators 4 \
#python ./cli-ibm-ddlrun.py \
#-p /gpfs/CompMicro/projects/nVidia_POC/dynamorph/microglia \
#-o /gpfs/CompMicro/projects/nVidia_POC/dynamorph/microglia/JUNE/nvidia-poc-4GPU-inf \
#-c 1

ddlrun \
-H cap \
-accelerators 4 \
python ./cli-ibm-ddlrun-horovod.py \
-p gpfs/CompMicro/projects/nVidia_POC/dynamorph/microglia \
-o /gpfs/CompMicro/projects/nVidia_POC/dynamorph/microglia/JUNE/nvidia-poc-horovod-4GPU \
-c 1