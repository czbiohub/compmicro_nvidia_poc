#!/bin/bash

# runs training using ibm's ddlrun with arguments



ddlrun --H cap \
--accelerators 4 \
--mpiarg -pami_noib \
python ./cli-ibm-ddlrun.py \
-p /gpfs/CompMicro/projects/nVidia_POC/dynamorph/microglia \
-o /gpfs/CompMicro/projects/nVidia_POC/dynamorph/microglia/JUNE/nvidia-poc-4GPU-ddlrun \
-c 1
