#!/bin/bash
#PJM -L rg=lecture-o
#PJM -g gt00
#PJM -L node=1
##PJM --mpi proc=1
#PJM --omp thread=48
#PJM -L elapse=00:15:00
#PJM -L jobenv=singularity
#PJM -j

export OS_NAME=Wisteria
export XOS_MMM_L_PAGING_POLICY=demand:demand:demand
cd /work/gt00/t00006/shell-model/cpp/build
cmake ../
make


# module load singularity aquarius
# SIF=/work/gt00/t00006/shell-model/singularity/sm-image.sif
# singularity exec --bind /work/gt00/t00006/shell-model:/workspaces/shell-model $SIF \
#     bash -c "cd /workspaces/shell-model/cpp/build && \
#              cmake -GNinja ../ && \
#              ninja"