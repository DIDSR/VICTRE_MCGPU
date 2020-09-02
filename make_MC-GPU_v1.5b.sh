#!/bin/bash
# 
#    ** Simple script to compile the MC-GPU code with CUDA, and optionally with MPI **
#
#      The installations paths to the CUDA toolkit and SDK (http://www.nvidia.com/cuda) and
#      the MPI library path may have to be adapted before runing the script.
#      The zlib.h library is used to allow gzip-ed input files.
# 
#      Please set the compilation architecture to the compute capability of your GPU cards!
#
#      Default paths:
#         CUDA:  /usr/local/cuda
#         SDK:   /usr/local/cuda/samples
#         MPI:   /usr/include/openmpi
#
# 
#                      @file    make_MC-GPU_v1.5b.sh
#                      @author  Andreu Badal [Andreu.Badal-Soler{at}fda.hhs.gov]
#                      @date    2020/09/01 
#   

# -- Compile GPU code with MPI:

echo " "
echo " -- Compiling MC-GPU v1.5 with CUDA and MPI:"
echo " "
echo "    Please set the compilation architecture to the compute capability of your GPU cards:"
echo "        - For NVIDIA GTX 2080:  -gencode=arch=compute_75,code=sm_75"
echo "        - For NVIDIA GTX 1080:  -gencode=arch=compute_61,code=sm_61"
echo "        - For NVIDIA GTX  780:  -gencode=arch=compute_50,code=sm_50"
echo " "
echo "    To run a simulation in 4 GPUs in parallel with openMPI in a single computer execute:"
echo "      $ time mpirun -n 4 ./MC-GPU_v1.5b.x MC-GPU_v1.5b.in | tee MC-GPU_v1.5b.out"
echo " "
echo "    To remove the undesirable openFabric warnings add the following mpirun option:  -mca btl tcp,sm,self" 
echo "    To run threads in multiple computers, save the code in a shared drive and use the options:  -x LD_LIBRARY_PATH -hostfile my_hostfile"
echo " "
echo " "
echo "    Running compilation command:"
echo "nvcc MC-GPU_v1.5b.cu -o MC-GPU_v1.5b.x -m64 -O3 -use_fast_math -DUSING_MPI -I. -I/usr/local/cuda/include -I/usr/local/cuda/samples/common/inc -I/usr/local/cuda/samples/shared/inc/ -I/usr/include/openmpi -L/usr/lib/ -lmpi -lz --ptxas-options=-v -gencode=arch=compute_50,code=sm_50 -gencode=arch=compute_61,code=sm_61"

nvcc MC-GPU_v1.5b.cu -o MC-GPU_v1.5b.x -m64 -O3 -use_fast_math -DUSING_MPI -I. -I/usr/local/cuda/include -I/usr/local/cuda/samples/common/inc -I/usr/local/cuda/samples/shared/inc/ -I/usr/include/openmpi -L/usr/lib/ -lmpi -lz --ptxas-options=-v -gencode=arch=compute_50,code=sm_50 -gencode=arch=compute_61,code=sm_61
# -gencode=arch=compute_75,code=sm_75


