#!/bin/bash
#PJM -L rg=lecture-o
#PJM -g gt00
#PJM -L node=2
##PJM --mpi proc=2
#PJM --omp thread=48
#PJM -L elapse=00:15:00
#PJM -L jobenv=singularity
#PJM -j

export XOS_MMM_L_PAGING_POLICY=demand:demand:demand
cd /work/gt00/t00006/shell-model/cpp
module load singularity fjmpi
SIF=/work/gt00/t00006/shell-model/singularity/sm-image.sif
mpiexec -n 2 singularity exec --bind /work/gt00/t00006/shell-model:/workspaces/shell-model $SIF \
            bash -c "cd /workspaces/shell-model/cpp/build && \
            ./exe && \
            echo 'end'"