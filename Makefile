
# ========================================================================================
#                                  MAKEFILE MC-GPU v1.3
#
# 
#   ** Simple script to compile the code MC-GPU v1.5b.
#
#      The installation paths to the CUDA toolkit and SDK (http://www.nvidia.com/cuda) 
#      and the MPI libraries (openMPI) may have to be modified by the user. 
#      The zlib.h library is used to allow gzip-ed input files.
#      The code can also be compiled for specific GPU architectures using the "-gencode=arch=compute_61,code=sm_61" option, 
#      where in this case 61 refers to compute capability 6.1.
#
#      Default paths:
#         CUDA:  /usr/local/cuda
#         SDK:   /usr/local/cuda/samples
#         MPI:   /usr/include/openmpi
#
# 
#                      @file    Makefile
#                      @author  Andreu Badal [Andreu.Badal-Soler (at) fda.hhs.gov]
#                      @date    2020/09/01
#   
# ========================================================================================

SHELL = /bin/sh

# Suffixes:
.SUFFIXES: .cu .o

# Compilers and linker:
CC = nvcc

# Program's name:
PROG = MC-GPU_v1.5b.x

# Include and library paths:
CUDA_PATH = /usr/local/cuda/include/
CUDA_LIB_PATH = /usr/local/cuda/lib64/
CUDA_SDK_PATH = /usr/local/cuda/samples/common/inc/
CUDA_SDK_LIB_PATH = /usr/local/cuda/samples/common/lib/linux/x86_64/
OPENMPI_PATH = /usr/include/openmpi


# Compiler's flags:
CFLAGS = -O3 -use_fast_math -m64 -DUSING_MPI -I./ -I$(CUDA_PATH) -I$(CUDA_SDK_PATH) -L$(CUDA_SDK_LIB_PATH) -L$(CUDA_LIB_PATH) -lcudart -lm -lz -I$(OPENMPI_PATH) -lmpi --ptxas-options=-v 
#  NOTE: you can compile the code for a specific GPU compute capability. For example, for compute capabilities 5.0 and 6.1, use flags:
#    -gencode=arch=compute_50,code=sm_50 -gencode=arch=compute_61,code=sm_61


# Command to erase files:
RM = /bin/rm -vf

# .cu files path:
SRCS = MC-GPU_v1.5b.cu

# Building the application:
default: $(PROG)
$(PROG):
	$(CC) $(CFLAGS) $(SRCS) -o $(PROG)

# Rule for cleaning re-compilable files
clean:
	$(RM) $(PROG)


