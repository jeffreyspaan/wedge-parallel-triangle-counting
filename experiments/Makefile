CFLAGS = -funroll-loops -O3

# CUDA
GENCODE_FLAGS =
NVCC = nvcc
# NVCC_FLAGS = -arch native -O3
# or set your own target generation:
NVCC_FLAGS = -gencode arch=compute_86,code=sm_86 -O3
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

all: edge_get tc_edge tc_vertex tc_base tc_reordered tc_spread tc_arrow tc_big_graphs tc_big_graphs_base tc_big_graphs_reordered tc_big_graphs_spread tc_big_graphs_edge tc_big_graphs_vertex

%.o: %.cu
	${NVCC} ${NVCC_FLAGS} -dc $< -o $@ $(NVCC_LIBS)

edge_get: edge_get.o
	${NVCC} ${NVCC_FLAGS} -Xcompiler "$(CFLAGS)" -o $@ edge_get.o $(CUDA_INC_DIR) $(CUDA_LIB_DIR) -lm $(CUDA_LINK_LIBS)

tc_edge: tc_edge.o
	${NVCC} ${NVCC_FLAGS} -Xcompiler "$(CFLAGS)" -o $@ tc_edge.o $(CUDA_INC_DIR) $(CUDA_LIB_DIR) -lm $(CUDA_LINK_LIBS)

tc_vertex: tc_vertex.o
	${NVCC} ${NVCC_FLAGS} -Xcompiler "$(CFLAGS)" -o $@ tc_vertex.o $(CUDA_INC_DIR) $(CUDA_LIB_DIR) -lm $(CUDA_LINK_LIBS)

tc_base: tc_base.o
	${NVCC} ${NVCC_FLAGS} -Xcompiler "$(CFLAGS)" -o $@ tc_base.o $(CUDA_INC_DIR) $(CUDA_LIB_DIR) -lm $(CUDA_LINK_LIBS)

tc_reordered: tc_reordered.o
	${NVCC} ${NVCC_FLAGS} -Xcompiler "$(CFLAGS)" -o $@ tc_reordered.o $(CUDA_INC_DIR) $(CUDA_LIB_DIR) -lm $(CUDA_LINK_LIBS)

tc_spread: tc_spread.o
	${NVCC} ${NVCC_FLAGS} -Xcompiler "$(CFLAGS)" -o $@ tc_spread.o $(CUDA_INC_DIR) $(CUDA_LIB_DIR) -lm $(CUDA_LINK_LIBS)

tc_arrow: tc_arrow.o
	${NVCC} ${NVCC_FLAGS} -Xcompiler "$(CFLAGS)" -o $@ tc_arrow.o $(CUDA_INC_DIR) $(CUDA_LIB_DIR) -lm $(CUDA_LINK_LIBS)

tc_big_graphs: tc_big_graphs.o
	${NVCC} ${NVCC_FLAGS} -Xcompiler "$(CFLAGS)" -o $@ tc_big_graphs.o $(CUDA_INC_DIR) $(CUDA_LIB_DIR) -lm $(CUDA_LINK_LIBS)

tc_big_graphs_base: tc_big_graphs_base.o
	${NVCC} ${NVCC_FLAGS} -Xcompiler "$(CFLAGS)" -o $@ tc_big_graphs_base.o $(CUDA_INC_DIR) $(CUDA_LIB_DIR) -lm $(CUDA_LINK_LIBS)

tc_big_graphs_reordered: tc_big_graphs_reordered.o
	${NVCC} ${NVCC_FLAGS} -Xcompiler "$(CFLAGS)" -o $@ tc_big_graphs_reordered.o $(CUDA_INC_DIR) $(CUDA_LIB_DIR) -lm $(CUDA_LINK_LIBS)

tc_big_graphs_spread: tc_big_graphs_spread.o
	${NVCC} ${NVCC_FLAGS} -Xcompiler "$(CFLAGS)" -o $@ tc_big_graphs_spread.o $(CUDA_INC_DIR) $(CUDA_LIB_DIR) -lm $(CUDA_LINK_LIBS)

tc_big_graphs_edge: tc_big_graphs_edge.o
	${NVCC} ${NVCC_FLAGS} -Xcompiler "$(CFLAGS)" -o $@ tc_big_graphs_edge.o $(CUDA_INC_DIR) $(CUDA_LIB_DIR) -lm $(CUDA_LINK_LIBS)

tc_big_graphs_vertex: tc_big_graphs_vertex.o
	${NVCC} ${NVCC_FLAGS} -Xcompiler "$(CFLAGS)" -o $@ tc_big_graphs_vertex.o $(CUDA_INC_DIR) $(CUDA_LIB_DIR) -lm $(CUDA_LINK_LIBS)

clean: 
	rm -f *~ $(OBJS_CUDA) edge_get tc_edge tc_vertex tc_base tc_reordered tc_spread tc_arrow tc_big_graphs tc_big_graphs_base tc_big_graphs_reordered tc_big_graphs_spread tc_big_graphs_edge tc_big_graphs_vertex