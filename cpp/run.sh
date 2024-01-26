#!/bin/bash
#PJM -L rg=lecture-o
#PJM -g gt00
#PJM -L node=12
#PJM --mpi proc=12
#PJM -L elapse=00:15:00
#PJM -L jobenv=singularity
#PJM -j

cd $PJM_0_WORKDIR
module load gcc cmake singularity
SIF=/work/gt00/t00006/shell-model/singularity/sm-image.file
singularity exec $SIF --bind `pwd` --bind /work/gt00/t00006/shell-model $SIF cd cpp/build && ./exe