#!/bin/bash
# 
#   ** Simple script to compile the code MC-GPU v1.5 with CUDA 5.0 **
#
#      The installations paths to the CUDA toolkit and SDK (http://www.nvidia.com/cuda) and the MPI 
#      library path may have to be adapted before runing the script!
#      The zlib.h library is used to allow gzip-ed input files.
# 
#      Default paths:
#         CUDA:  /usr/local/cuda
#         SDK:   /usr/local/cuda/samples
#         MPI:   /usr/include/openmpi
#
# 
#                      @file    make_MC-GPU_v1.5.sh
#                      @author  Andreu Badal [Andreu.Badal-Soler(at)fda.hhs.gov]
#                      @date    2017/06/28 
#                               2012/12/12
#   

# -- Compile GPU code with MPI:

echo " "
echo " -- Compiling MC-GPU v1.5 with MPI:"
echo " "
echo "    To compile for NVIDIA GTX 1080, use option: -gencode=arch=compute_61,code=sm_61"
echo " "
echo "    To run a simulation in parallel with openMPI execute:"
echo "      $ time mpirun --tag-output -v -x LD_LIBRARY_PATH -hostfile hostfile_gpunodes -n 22 /GPU_cluster/MC-GPU_v1.4_DBT.x /GPU_cluster/MC-GPU_v1.4_DBT.in | tee MC-GPU_v1.4_DBT.out"
echo " "
echo "nvcc MC-GPU_v1.5b.cu -o MC-GPU_v1.5b.x -m64 -O3 -use_fast_math -DUSING_CUDA -DUSING_MPI -I. -I/usr/local/cuda/include -I/usr/local/cuda/samples/common/inc -I/usr/local/cuda/samples/shared/inc/ -I/usr/include/openmpi -L/usr/lib/ -lmpi -lz --ptxas-options=-v -gencode=arch=compute_61,code=sm_61 -gencode=arch=compute_50,code=sm_50 -gencode=arch=compute_30,code=sm_30"
nvcc MC-GPU_v1.5b.cu -o MC-GPU_v1.5b.x -m64 -O3 -use_fast_math -DUSING_CUDA -DUSING_MPI -I. -I/usr/local/cuda/include -I/usr/local/cuda/samples/common/inc -I/usr/local/cuda/samples/shared/inc/ -I/usr/include/openmpi -L/usr/lib/ -lmpi -lz --ptxas-options=-v -gencode=arch=compute_50,code=sm_50 -gencode=arch=compute_30,code=sm_30 


# FOR GTX 1080:   -gencode=arch=compute_61,code=sm_61 

# To disable the binary tree canonicalization, add compilation instruction "-DDISABLE_CANON"


# -- CPU compilation:
 
# ** GCC (with MPI):
# gcc -x c -DUSING_MPI MC-GPU_v1.4_DBT.cu -o MC-GPU_v1.4_DBT_gcc_MPI.x -Wall -O3 -ffast-math -ftree-vectorize -ftree-vectorizer-verbose=1 -funroll-loops -static-libgcc -I./ -lm -I/usr/include/openmpi -I/usr/lib/openmpi/include/openmpi/ -L/usr/lib/openmpi/lib -lmpi

     
# ** Intel compiler (with MPI):
# icc -x c -O3 -ipo -no-prec-div -msse4.2 -parallel -Wall -DUSING_MPI MC-GPU_v1.4_DBT.cu -o MC-GPU_v1.4_DBT_icc_MPI.x -I./ -lm -I/usr/include/openmpi -L/usr/lib/openmpi/lib/ -lmpi


# ** PGI compiler:
# pgcc -fast,sse -O3 -Mipa=fast -Minfo -csuffix=cu -Mconcur MC-GPU_v1.4_DBT.cu -I./ -lm -o MC-GPU_v1.4_DBT_PGI.x

