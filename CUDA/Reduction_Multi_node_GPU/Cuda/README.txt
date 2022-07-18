module load cuda/11.3
nvcc \
-I/work/HPC_SYS/twnia2/openmpi/openmpi-4.1.1-ucx-1.10.1-cuda-11.3/include \
-L/work/HPC_SYS/twnia2/openmpi/openmpi-4.1.1-ucx-1.10.1-cuda-11.3/lib \
-lmpi multi_node.cu \
-o a.out
