
# GCC
CC     = gcc
CFLAGS = -funroll-loops -O3

# CUDA
GENCODE_FLAGS =
NVCC = nvcc
NVCC_FLAGS = -arch native -O3
# or set your own target generation:
# NVCC_FLAGS = -gencode arch=compute_86,code=sm_86 -O3
NVCC_LIBS=

# CUDA directory:
CUDA_ROOT_DIR = /usr/local/cuda
# CUDA library directory:
CUDA_LIB_DIR = -L$(CUDA_ROOT_DIR)/lib64
# CUDA include directory:
CUDA_INC_DIR = -I$(CUDA_ROOT_DIR)/include
# CUDA linking libraries:
CUDA_LINK_LIBS = -lcudart

SRCS_CUDA = $(wildcard *.cu)
OBJS_CUDA = $(SRCS_CUDA:.cu=.o)

all: tc

%.o: %.cu
	${NVCC} ${GPU} ${NVCC_FLAGS} -dc $< -o $@ $(NVCC_LIBS)

tc: $(OBJS_CUDA)
	${NVCC} ${NVCC_FLAGS} -Xcompiler "$(CFLAGS)" -o $@ $(OBJS_CUDA) $(CUDA_INC_DIR) $(CUDA_LIB_DIR) -lm $(CUDA_LINK_LIBS)

clean: 
	rm -f *~ $(OBJS_CUDA) tc
