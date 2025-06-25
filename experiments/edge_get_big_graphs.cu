/* Edge retrieval strategy experiment for big graphs
 * Jeffrey Spaan, Kuan-Hsun Chen, David Bader, Ana-Lucia Varbanescu
 *
 * Built on the work and code of David Bader. See https://github.com/Bader-Research/triangle-counting/ and https://doi.org/10.1109/HPEC58863.2023.10363539
 *
 * See usage() for instructions.
 * 
 * Assumptions:
 *	- Target GPU is device 0.
 *	- Number of vertices < (uint32_max / 2).
 *	- Number of edges < (uint64_max / 2).
 * 	- Max degree < uint32_max
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <sys/time.h>

#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_segmented_sort.cuh>

#define CHECK_BOUNDS 1
#define RESET_DEVICE 0

#define BINSEARCH_CONSTANT_LEVELS 12
#define BINSEARCH_CONSTANT_CACHE_SIZE ((1 << BINSEARCH_CONSTANT_LEVELS) - 1) // 2^levels - 1

#define UINT_t uint32_t
#define INT_t int32_t
#define ULONG_t uint64_t

#define max2(a,b) ((a)>(b)?(a):(b))
#define min2(a,b) ((a)<(b)?(a):(b))

static struct timeval	tp;
static struct timezone tzp;

#define get_seconds()	 (gettimeofday(&tp, &tzp), \
												(double)tp.tv_sec + (double)tp.tv_usec / 1000000.0)

#define checkCudaErrors(call)																						\
	do {																																	\
		cudaError_t err = call;																	 						\
		if (err != cudaSuccess) {																 						\
			fprintf(stderr, "CUDA error at %s %d: %s\n", __FILE__, __LINE__,	\
						 cudaGetErrorString(err));																	\
			exit(EXIT_FAILURE);																		 						\
		}																												 						\
	} while (0)

enum preprocess_t { PREPROCESS_CPU = 0, PREPROCESS_GPU, PREPROCESS_GPU_CONSTRAINED};

typedef struct {
	UINT_t numVertices;
	ULONG_t numEdges;
	ULONG_t* rowPtr;
	UINT_t* colInd;
} BIG_GRAPH_TYPE;

typedef struct {
	UINT_t src;
	UINT_t dst;
} edge_t;

typedef struct {
	UINT_t id;
	UINT_t *edges;
	UINT_t num_edges;
} preprocess_vertex_t;

typedef struct {
	double copy;
	double exec;
} GPU_time;

/*********
 *  GPU  *
 *********/

__constant__ ULONG_t c_binary_search_cache[BINSEARCH_CONSTANT_CACHE_SIZE];

__device__ INT_t binary_search_ULONG_GPU(const UINT_t* list, const ULONG_t start, const ULONG_t end, const UINT_t target) {
	ULONG_t s=start, e=end, mid;
	while (s < e) {
		mid = (s + e) >> 1;
		if (list[mid] == target)
			return mid;

		if (list[mid] < target)
			s = mid + 1;
		else
			e = mid;
	}
	return -1;
}

__device__ UINT_t binary_search_closest_ULONG_GPU(const ULONG_t* list, const UINT_t start, const UINT_t end, const ULONG_t target) {
	/* Finds the index of the rightmost closest value smaller or equal than target, e.g.,
	 * for target 1 and list=[0,0,0,2,2,2] it returns 2,
	 * for target 2 and list=[0,0,0,2,2,2] it returns 5.
	 * Assumes list[0]=0
	 * Assumes end-1 <= UINT_MAX/2
	 */

	UINT_t s=start, e=end, mid;
	while (s < e) {
		mid = (s + e) >> 1;

		if (list[mid] < target+1) {
			s = mid + 1;
		} else {
			e = mid;
		}
	}
	
	return max2(start, (s > 0) ? s-1 : 0);
}

__device__ UINT_t binary_search_closest_ULONG_constant_GPU(const ULONG_t *list, const UINT_t start, const UINT_t end, const ULONG_t target) {
	/* Finds the index of the rightmost closest value smaller or equal than target.
	 * Uses constant memory for the first BINSEARCH_CONSTANT_LEVELS levels.
	 */
	ULONG_t mid;

	UINT_t g_s = start;
	UINT_t g_e = end;
	UINT_t g_mid;

	UINT_t c_index = 0;

	#pragma unroll
	for (UINT_t iter=0; iter<BINSEARCH_CONSTANT_LEVELS; iter++) {
		mid = c_binary_search_cache[c_index];
		g_mid = (g_s+g_e) >> 1;

		c_index *= 2;
		c_index += 1;

		if (mid < target+1) {
			c_index += 1;
			g_s = g_mid+1;
		} else {
			g_e = g_mid;
		}
	}

	g_s = max2(start, (g_s > 0) ? g_s-1 : 0);
	return binary_search_closest_ULONG_GPU(list, g_s, g_e, target);
}

typedef struct {
	UINT_t src;
} edge_src_t;

__global__ void edge_get_edgelist_GPU_kernel(const ULONG_t *g_Ap, const UINT_t *g_Ai, const edge_t *g_edges, const UINT_t num_vertices, const ULONG_t edges_start, const ULONG_t edges_stop, UINT_t *g_out) {
	const ULONG_t tid = ((ULONG_t) (blockIdx.x * blockDim.x + threadIdx.x)) + edges_start;
	
	if (tid < edges_stop) {
		UINT_t v = g_edges[tid].src;
		UINT_t w = g_edges[tid].dst;

		ULONG_t wb = g_Ap[w];
		ULONG_t we = g_Ap[w+1];

		if (binary_search_ULONG_GPU(g_Ai, wb, we, v) >= 0) {
			g_out[tid] = 1;
		} else {
			g_out[tid] = 0;
		}
	}
}

__global__ void edge_get_edgelist_src_only_GPU_kernel(const ULONG_t *g_Ap, const UINT_t *g_Ai, const edge_src_t *g_edges_src, const UINT_t num_vertices, const ULONG_t edges_start, const ULONG_t edges_stop, UINT_t *g_out) {
	const ULONG_t tid = ((ULONG_t) (blockIdx.x * blockDim.x + threadIdx.x)) + edges_start;
	
	if (tid < edges_stop) {
		UINT_t v = g_edges_src[tid].src;
		UINT_t w = g_Ai[tid];

		ULONG_t wb = g_Ap[w];
		ULONG_t we = g_Ap[w+1];

		if (binary_search_ULONG_GPU(g_Ai, wb, we, v) >= 0) {
			g_out[tid] = 1;
		} else {
			g_out[tid] = 0;
		}
	}
}

__global__ void edge_get_binary_search_GPU_kernel(const ULONG_t *g_Ap, const UINT_t *g_Ai, const UINT_t num_vertices, const ULONG_t edges_start, const ULONG_t edges_stop, UINT_t *g_out) {
	const ULONG_t tid = ((ULONG_t) (blockIdx.x * blockDim.x + threadIdx.x)) + edges_start;
	
	if (tid < edges_stop) {
		UINT_t w = g_Ai[tid];
		UINT_t v = binary_search_closest_ULONG_GPU(g_Ap, 0, num_vertices, tid);

		ULONG_t wb = g_Ap[w];
		ULONG_t we = g_Ap[w+1];

		if (binary_search_ULONG_GPU(g_Ai, wb, we, v) >= 0) {
			g_out[tid] = 1;
		} else {
			g_out[tid] = 0;
		}
	}
}

__global__ void edge_get_binary_search_cached_GPU_kernel(const ULONG_t *g_Ap, const UINT_t *g_Ai, const UINT_t num_vertices, const ULONG_t edges_start, const ULONG_t edges_stop, UINT_t *g_out) {
	const ULONG_t tid = ((ULONG_t) (blockIdx.x * blockDim.x + threadIdx.x)) + edges_start;
	
	if (tid < edges_stop) {
		UINT_t w = g_Ai[tid];
		UINT_t v = binary_search_closest_ULONG_constant_GPU(g_Ap, 0, num_vertices, tid);

		ULONG_t wb = g_Ap[w];
		ULONG_t we = g_Ap[w+1];

		if (binary_search_ULONG_GPU(g_Ai, wb, we, v) >= 0) {
			g_out[tid] = 1;
		} else {
			g_out[tid] = 0;
		}
	}
}

/*********
 *  CPU  *
 *********/

static void assert_malloc(const void *ptr) {
	if (ptr==NULL) {
		fprintf(stderr,"ERROR: failed to allocate host memory.\n");
		exit(EXIT_FAILURE);
	}
}

void build_binary_search_cache(ULONG_t *src, ULONG_t *cache, UINT_t level, UINT_t max_level, UINT_t i, UINT_t s, UINT_t e) {
	if (level < max_level) {
		UINT_t mid = (s + e) / 2;
		cache[i] = src[mid];
		build_binary_search_cache(src, cache, level+1, max_level, i*2 + 1, s, mid);
		build_binary_search_cache(src, cache, level+1, max_level, i*2 + 2, mid+1, e);
	}
}

void edge_get_edgelist_GPU(const BIG_GRAPH_TYPE *graph, GPU_time *t) {
	ULONG_t *d_Ap;
	UINT_t *d_Ai;
	edge_t *d_edges;
	UINT_t *d_out;

	cudaEvent_t GPU_copy_start, GPU_copy_stop, GPU_exec_start, GPU_exec_stop;
	float GPU_copy_elapsed, GPU_exec_elapsed;
	checkCudaErrors(cudaEventCreate(&GPU_copy_start));
	checkCudaErrors(cudaEventCreate(&GPU_copy_stop));
	checkCudaErrors(cudaEventCreate(&GPU_exec_start));
	checkCudaErrors(cudaEventCreate(&GPU_exec_stop));

	edge_t *h_edges = (edge_t *) malloc(graph->numEdges * sizeof(edge_t));
	ULONG_t edge_ctr = 0;

	for (UINT_t v=0; v<graph->numVertices; v++) {
		for (ULONG_t i=graph->rowPtr[v]; i<graph->rowPtr[v+1]; i++) {
			UINT_t w = graph->colInd[i];
			h_edges[edge_ctr].src = v;
			h_edges[edge_ctr].dst = w;
			edge_ctr++;
		}
	}

	checkCudaErrors(cudaEventRecord(GPU_copy_start));

	checkCudaErrors(cudaMalloc((void **)&d_Ap, (graph->numVertices + 1) * sizeof(ULONG_t)));
	checkCudaErrors(cudaMalloc((void **)&d_Ai, graph->numEdges * sizeof(UINT_t)));
	checkCudaErrors(cudaMalloc((void **)&d_edges, graph->numEdges * sizeof(edge_t)));
	checkCudaErrors(cudaMalloc((void **)&d_out, graph->numEdges * sizeof(UINT_t)));

	checkCudaErrors(cudaMemcpy(d_Ap, graph->rowPtr, (graph->numVertices + 1) * sizeof(ULONG_t), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_Ai, graph->colInd, graph->numEdges * sizeof(UINT_t), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_edges, h_edges, graph->numEdges * sizeof(edge_t), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaEventRecord(GPU_copy_stop));
	checkCudaErrors(cudaEventSynchronize(GPU_copy_stop));
	checkCudaErrors(cudaEventElapsedTime(&GPU_copy_elapsed, GPU_copy_start, GPU_copy_stop));
	t->copy += GPU_copy_elapsed;

	UINT_t num_threads = 128;
	ULONG_t max_blocks = (((ULONG_t) 1 << 31)-1);
	ULONG_t max_edges = max_blocks * num_threads; // max edges in a single launch

	for (ULONG_t start=0; start<graph->numEdges; start+=max_edges) {
		ULONG_t stop = min2(start + max_edges, graph->numEdges);
		ULONG_t num_blocks = ((stop - start) / num_threads);

		if (num_blocks < max_blocks)
			num_blocks++;

		dim3 grid(num_blocks, 1, 1);
		dim3 threads(num_threads, 1, 1);

		checkCudaErrors(cudaEventRecord(GPU_exec_start));

		edge_get_edgelist_GPU_kernel<<<grid, threads>>>(d_Ap, d_Ai, d_edges, graph->numVertices, start, stop, d_out);

		checkCudaErrors(cudaEventRecord(GPU_exec_stop));
		checkCudaErrors(cudaEventSynchronize(GPU_exec_stop));
		checkCudaErrors(cudaEventElapsedTime(&GPU_exec_elapsed, GPU_exec_start, GPU_exec_stop));
		t->exec += GPU_exec_elapsed;
	}

	UINT_t *h_out = (UINT_t *) malloc(graph->numEdges * sizeof(UINT_t));
	checkCudaErrors(cudaMemcpy(h_out, d_out, graph->numEdges * sizeof(UINT_t), cudaMemcpyDeviceToHost));

	/* Confirm (w,v) does not exist (=0) for every edge (v,w). */
	for (ULONG_t i=0; i<graph->numEdges; i++) {
		if (h_out[i] != 0) {
			fprintf(stderr, "Nonzero found.\n");
			exit(EXIT_FAILURE);
		}
	}

	checkCudaErrors(cudaFree(d_Ap));
	checkCudaErrors(cudaFree(d_Ai));
	checkCudaErrors(cudaFree(d_edges));
	checkCudaErrors(cudaFree(d_out));

	checkCudaErrors(cudaEventDestroy(GPU_copy_start));
	checkCudaErrors(cudaEventDestroy(GPU_copy_stop));
	checkCudaErrors(cudaEventDestroy(GPU_exec_start));
	checkCudaErrors(cudaEventDestroy(GPU_exec_stop));

	free(h_edges);
	free(h_out);
	
#if RESET_DEVICE
	checkCudaErrors(cudaDeviceReset());
#endif
}

void edge_get_edgelist_src_only_GPU(const BIG_GRAPH_TYPE *graph, GPU_time *t) {
	ULONG_t *d_Ap;
	UINT_t *d_Ai;
	edge_src_t *d_edges_src;
	UINT_t *d_out;

	cudaEvent_t GPU_copy_start, GPU_copy_stop, GPU_exec_start, GPU_exec_stop;
	float GPU_copy_elapsed, GPU_exec_elapsed;
	checkCudaErrors(cudaEventCreate(&GPU_copy_start));
	checkCudaErrors(cudaEventCreate(&GPU_copy_stop));
	checkCudaErrors(cudaEventCreate(&GPU_exec_start));
	checkCudaErrors(cudaEventCreate(&GPU_exec_stop));

	edge_src_t *h_edges_src = (edge_src_t *) malloc(graph->numEdges * sizeof(edge_src_t));
	ULONG_t edge_ctr = 0;

	for (UINT_t v=0; v<graph->numVertices; v++) {
		for (ULONG_t i=graph->rowPtr[v]; i<graph->rowPtr[v+1]; i++) {
			h_edges_src[edge_ctr].src = v;
			edge_ctr++;
		}
	}

	checkCudaErrors(cudaEventRecord(GPU_copy_start));

	checkCudaErrors(cudaMalloc((void **)&d_Ap, (graph->numVertices + 1) * sizeof(ULONG_t)));
	checkCudaErrors(cudaMalloc((void **)&d_Ai, graph->numEdges * sizeof(UINT_t)));
	checkCudaErrors(cudaMalloc((void **)&d_edges_src, graph->numEdges * sizeof(edge_src_t)));
	checkCudaErrors(cudaMalloc((void **)&d_out, graph->numEdges * sizeof(UINT_t)));

	checkCudaErrors(cudaMemcpy(d_Ap, graph->rowPtr, (graph->numVertices + 1) * sizeof(ULONG_t), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_Ai, graph->colInd, graph->numEdges * sizeof(UINT_t), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_edges_src, h_edges_src, graph->numEdges * sizeof(edge_src_t), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaEventRecord(GPU_copy_stop));
	checkCudaErrors(cudaEventSynchronize(GPU_copy_stop));
	checkCudaErrors(cudaEventElapsedTime(&GPU_copy_elapsed, GPU_copy_start, GPU_copy_stop));
	t->copy += GPU_copy_elapsed;

	UINT_t num_threads = 128;
	ULONG_t max_blocks = (((ULONG_t) 1 << 31)-1);
	ULONG_t max_edges = max_blocks * num_threads; // max edges in a single launch

	for (ULONG_t start=0; start<graph->numEdges; start+=max_edges) {
		ULONG_t stop = min2(start + max_edges, graph->numEdges);
		ULONG_t num_blocks = ((stop - start) / num_threads);

		if (num_blocks < max_blocks)
			num_blocks++;

		dim3 grid(num_blocks, 1, 1);
		dim3 threads(num_threads, 1, 1);

		checkCudaErrors(cudaEventRecord(GPU_exec_start));

		edge_get_edgelist_src_only_GPU_kernel<<<grid, threads>>>(d_Ap, d_Ai, d_edges_src, graph->numVertices, start, stop, d_out);

		checkCudaErrors(cudaEventRecord(GPU_exec_stop));
		checkCudaErrors(cudaEventSynchronize(GPU_exec_stop));
		checkCudaErrors(cudaEventElapsedTime(&GPU_exec_elapsed, GPU_exec_start, GPU_exec_stop));
		t->exec += GPU_exec_elapsed;
	}
	
	UINT_t *h_out = (UINT_t *) malloc(graph->numEdges * sizeof(UINT_t));
	checkCudaErrors(cudaMemcpy(h_out, d_out, graph->numEdges * sizeof(UINT_t), cudaMemcpyDeviceToHost));

	/* Confirm (w,v) does not exist (=0) for every edge (v,w). */
	for (ULONG_t i=0; i<graph->numEdges; i++) {
		if (h_out[i] != 0) {
			fprintf(stderr, "Nonzero found.\n");
			exit(EXIT_FAILURE);
		}
	}

	checkCudaErrors(cudaFree(d_Ap));
	checkCudaErrors(cudaFree(d_Ai));
	checkCudaErrors(cudaFree(d_edges_src));
	checkCudaErrors(cudaFree(d_out));

	checkCudaErrors(cudaEventDestroy(GPU_copy_start));
	checkCudaErrors(cudaEventDestroy(GPU_copy_stop));
	checkCudaErrors(cudaEventDestroy(GPU_exec_start));
	checkCudaErrors(cudaEventDestroy(GPU_exec_stop));

	free(h_edges_src);
	free(h_out);
	
#if RESET_DEVICE
	checkCudaErrors(cudaDeviceReset());
#endif
}

void edge_get_binary_search_GPU(const BIG_GRAPH_TYPE *graph, GPU_time *t) {
	ULONG_t *d_Ap;
	UINT_t *d_Ai;
	UINT_t *d_out;

	cudaEvent_t GPU_copy_start, GPU_copy_stop, GPU_exec_start, GPU_exec_stop;
	float GPU_copy_elapsed, GPU_exec_elapsed;
	checkCudaErrors(cudaEventCreate(&GPU_copy_start));
	checkCudaErrors(cudaEventCreate(&GPU_copy_stop));
	checkCudaErrors(cudaEventCreate(&GPU_exec_start));
	checkCudaErrors(cudaEventCreate(&GPU_exec_stop));

	checkCudaErrors(cudaEventRecord(GPU_copy_start));

	checkCudaErrors(cudaMalloc((void **)&d_Ap, (graph->numVertices + 1) * sizeof(ULONG_t)));
	checkCudaErrors(cudaMalloc((void **)&d_Ai, graph->numEdges * sizeof(UINT_t)));
	checkCudaErrors(cudaMalloc((void **)&d_out, graph->numEdges * sizeof(UINT_t)));

	checkCudaErrors(cudaMemcpy(d_Ap, graph->rowPtr, (graph->numVertices + 1) * sizeof(ULONG_t), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_Ai, graph->colInd, graph->numEdges * sizeof(UINT_t), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaEventRecord(GPU_copy_stop));
	checkCudaErrors(cudaEventSynchronize(GPU_copy_stop));
	checkCudaErrors(cudaEventElapsedTime(&GPU_copy_elapsed, GPU_copy_start, GPU_copy_stop));
	t->copy += GPU_copy_elapsed;

	UINT_t num_threads = 128;
	ULONG_t max_blocks = (((ULONG_t) 1 << 31)-1);
	ULONG_t max_edges = max_blocks * num_threads; // max edges in a single launch

	for (ULONG_t start=0; start<graph->numEdges; start+=max_edges) {
		ULONG_t stop = min2(start + max_edges, graph->numEdges);
		ULONG_t num_blocks = ((stop - start) / num_threads);

		if (num_blocks < max_blocks)
			num_blocks++;

		dim3 grid(num_blocks, 1, 1);
		dim3 threads(num_threads, 1, 1);

		checkCudaErrors(cudaEventRecord(GPU_exec_start));

		edge_get_binary_search_GPU_kernel<<<grid, threads>>>(d_Ap, d_Ai, graph->numVertices, start, stop, d_out);

		checkCudaErrors(cudaEventRecord(GPU_exec_stop));
		checkCudaErrors(cudaEventSynchronize(GPU_exec_stop));
		checkCudaErrors(cudaEventElapsedTime(&GPU_exec_elapsed, GPU_exec_start, GPU_exec_stop));
		t->exec += GPU_exec_elapsed;
	}

	UINT_t *h_out = (UINT_t *) malloc(graph->numEdges * sizeof(UINT_t));
	checkCudaErrors(cudaMemcpy(h_out, d_out, graph->numEdges * sizeof(UINT_t), cudaMemcpyDeviceToHost));

	/* Confirm (w,v) does not exist (=0) for every edge (v,w). */
	for (ULONG_t i=0; i<graph->numEdges; i++) {
		if (h_out[i] != 0) {
			fprintf(stderr, "Nonzero found.\n");
			exit(EXIT_FAILURE);
		}
	}

	checkCudaErrors(cudaFree(d_Ap));
	checkCudaErrors(cudaFree(d_Ai));
	checkCudaErrors(cudaFree(d_out));

	checkCudaErrors(cudaEventDestroy(GPU_copy_start));
	checkCudaErrors(cudaEventDestroy(GPU_copy_stop));
	checkCudaErrors(cudaEventDestroy(GPU_exec_start));
	checkCudaErrors(cudaEventDestroy(GPU_exec_stop));

	free(h_out);

#if RESET_DEVICE
	checkCudaErrors(cudaDeviceReset());
#endif
}

void edge_get_binary_search_cached_GPU(const BIG_GRAPH_TYPE *graph, GPU_time *t) {
	ULONG_t *d_Ap;
	UINT_t *d_Ai;
	UINT_t *d_out;

	cudaEvent_t GPU_copy_start, GPU_copy_stop, GPU_exec_start, GPU_exec_stop;
	float GPU_copy_elapsed, GPU_exec_elapsed;
	checkCudaErrors(cudaEventCreate(&GPU_copy_start));
	checkCudaErrors(cudaEventCreate(&GPU_copy_stop));
	checkCudaErrors(cudaEventCreate(&GPU_exec_start));
	checkCudaErrors(cudaEventCreate(&GPU_exec_stop));
	
	ULONG_t *h_rowPtr_cache = (ULONG_t *) malloc(BINSEARCH_CONSTANT_CACHE_SIZE * sizeof(ULONG_t));
	assert_malloc(h_rowPtr_cache);
	build_binary_search_cache(graph->rowPtr, h_rowPtr_cache, 0, BINSEARCH_CONSTANT_LEVELS, 0, 0, graph->numVertices);

	checkCudaErrors(cudaEventRecord(GPU_copy_start));

	checkCudaErrors(cudaMalloc((void **)&d_Ap, (graph->numVertices + 1) * sizeof(ULONG_t)));
	checkCudaErrors(cudaMalloc((void **)&d_Ai, graph->numEdges * sizeof(UINT_t)));
	checkCudaErrors(cudaMalloc((void **)&d_out, graph->numEdges * sizeof(UINT_t)));

	checkCudaErrors(cudaMemcpy(d_Ap, graph->rowPtr, (graph->numVertices + 1) * sizeof(ULONG_t), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_Ai, graph->colInd, graph->numEdges * sizeof(UINT_t), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpyToSymbol(c_binary_search_cache, h_rowPtr_cache, BINSEARCH_CONSTANT_CACHE_SIZE * sizeof(ULONG_t)));

	checkCudaErrors(cudaEventRecord(GPU_copy_stop));
	checkCudaErrors(cudaEventSynchronize(GPU_copy_stop));
	checkCudaErrors(cudaEventElapsedTime(&GPU_copy_elapsed, GPU_copy_start, GPU_copy_stop));
	t->copy += GPU_copy_elapsed;

	UINT_t num_threads = 128;
	ULONG_t max_blocks = (((ULONG_t) 1 << 31)-1);
	ULONG_t max_edges = max_blocks * num_threads; // max edges in a single launch

	for (ULONG_t start=0; start<graph->numEdges; start+=max_edges) {
		ULONG_t stop = min2(start + max_edges, graph->numEdges);
		ULONG_t num_blocks = ((stop - start) / num_threads);

		if (num_blocks < max_blocks)
			num_blocks++;

		dim3 grid(num_blocks, 1, 1);
		dim3 threads(num_threads, 1, 1);

		checkCudaErrors(cudaEventRecord(GPU_exec_start));

		edge_get_binary_search_cached_GPU_kernel<<<grid, threads>>>(d_Ap, d_Ai, graph->numVertices, start, stop, d_out);

		checkCudaErrors(cudaEventRecord(GPU_exec_stop));
		checkCudaErrors(cudaEventSynchronize(GPU_exec_stop));
		checkCudaErrors(cudaEventElapsedTime(&GPU_exec_elapsed, GPU_exec_start, GPU_exec_stop));
		t->exec += GPU_exec_elapsed;
	}
	
	UINT_t *h_out = (UINT_t *) malloc(graph->numEdges * sizeof(UINT_t));
	checkCudaErrors(cudaMemcpy(h_out, d_out, graph->numEdges * sizeof(UINT_t), cudaMemcpyDeviceToHost));

	/* Confirm (w,v) does not exist (=0) for every edge (v,w). */
	for (ULONG_t i=0; i<graph->numEdges; i++) {
		if (h_out[i] != 0) {
			fprintf(stderr, "Nonzero found.\n");
			exit(EXIT_FAILURE);
		}
	}

	checkCudaErrors(cudaFree(d_Ap));
	checkCudaErrors(cudaFree(d_Ai));
	checkCudaErrors(cudaFree(d_out));

	checkCudaErrors(cudaEventDestroy(GPU_copy_start));
	checkCudaErrors(cudaEventDestroy(GPU_copy_stop));
	checkCudaErrors(cudaEventDestroy(GPU_exec_start));
	checkCudaErrors(cudaEventDestroy(GPU_exec_stop));

	free(h_rowPtr_cache);
	free(h_out);

#if RESET_DEVICE
	checkCudaErrors(cudaDeviceReset());
#endif
}

void usage() {
	printf("Edge retrieval strategy experiment\n\n");
	printf("Usage:\n\n");
	printf("Either one of these must be selected:\n");
	printf(" -m <filename>        [Input graph in Matrix Market format]\n");
	printf(" -e <filename>        [Input graph in edge list format]\n");
	printf("Optional arguments:\n");
	printf(" -l <num>             [Loop count]\n");
	printf(" -z                   [Input graph is zero-indexed]\n");
	printf(" -p                   [Preprocessing style, 0:CPU, 1:GPU, 2:GPU low-memory (default)]\n");
	printf("\n");
	printf("Example:\n");
	printf("./edge_get -m ../Amazon0302.mtx -l 10\n");
	exit(EXIT_FAILURE);
}

static int compareInt_t(const void *a, const void *b) {
	UINT_t arg1 = *(const UINT_t *)a;
	UINT_t arg2 = *(const UINT_t *)b;
	if (arg1 < arg2) return -1;
	if (arg1 > arg2) return 1;
	return 0;
}

static int compareEdge_t(const void *a, const void *b) {
	edge_t arg1 = *(const edge_t *) a;
	edge_t arg2 = *(const edge_t *) b;
	if (arg1.src < arg2.src) return -1;
	if (arg1.src > arg2.src) return 1;
	if ((arg1.src == arg2.src) && (arg1.dst < arg2.dst)) return -1;
	if ((arg1.src == arg2.src) && (arg1.dst > arg2.dst)) return 1;
	return 0;
}

static int compare_vertex_degree_ascending(const void *a, const void *b) {
	preprocess_vertex_t arg1 = *(const preprocess_vertex_t *) a;
	preprocess_vertex_t arg2 = *(const preprocess_vertex_t *) b;
	if (arg1.num_edges < arg2.num_edges) return -1;
	if (arg1.num_edges > arg2.num_edges) return 1;
	return 0;
}

struct edge_decomposer_t {
  __host__ __device__ ::cuda::std::tuple<unsigned int&, unsigned int&> operator()(edge_t& key) const {
    return {key.src, key.dst};
  }
};

struct preprocess_vertex_decomposer_t {
  __host__ __device__ ::cuda::std::tuple<unsigned int&> operator()(preprocess_vertex_t& key) const {
    return {key.num_edges};
  }
};

edge_t *sort_edges_GPU(edge_t *d_in, edge_t *d_out, const UINT_t num_edges, bool use_double_buffer) {
	std::uint8_t* d_temp_storage{};
	std::size_t temp_storage_bytes{};

	if (use_double_buffer) {
		cub::DoubleBuffer<edge_t> d_keys(d_in, d_out);
		cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_keys, num_edges, edge_decomposer_t{});
		checkCudaErrors(cudaMalloc((void **) &d_temp_storage, temp_storage_bytes * sizeof(std::uint8_t)));
		cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_keys, num_edges, edge_decomposer_t{});
		checkCudaErrors(cudaFree(d_temp_storage));
		return d_keys.Current();
	} else {
		cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_in, d_out, num_edges, edge_decomposer_t{});
		checkCudaErrors(cudaMalloc((void **) &d_temp_storage, temp_storage_bytes * sizeof(std::uint8_t)));
		cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_in, d_out, num_edges, edge_decomposer_t{});
		checkCudaErrors(cudaFree(d_temp_storage));
		return d_out;
	}
}

preprocess_vertex_t *sort_vertices_GPU(preprocess_vertex_t *d_in, preprocess_vertex_t *d_out, const UINT_t num_vertices, bool use_double_buffer) {
	std::uint8_t* d_temp_storage{};
	std::size_t temp_storage_bytes{};

	if (use_double_buffer) {
		cub::DoubleBuffer<preprocess_vertex_t> d_keys(d_in, d_out);
		cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_keys, num_vertices, preprocess_vertex_decomposer_t{});
		checkCudaErrors(cudaMalloc((void **)&d_temp_storage, temp_storage_bytes * sizeof(std::uint8_t)));
		cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_keys, num_vertices, preprocess_vertex_decomposer_t{});
		checkCudaErrors(cudaFree(d_temp_storage));
		return d_keys.Current();
	} else {
		cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_in, d_out, num_vertices, preprocess_vertex_decomposer_t{});
		checkCudaErrors(cudaMalloc((void **)&d_temp_storage, temp_storage_bytes * sizeof(std::uint8_t)));
		cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_in, d_out, num_vertices, preprocess_vertex_decomposer_t{});
		checkCudaErrors(cudaFree(d_temp_storage));
		return d_out;
	}
}

UINT_t *sort_colInd_GPU(ULONG_t *d_rowPtr, UINT_t *d_colInd_in, UINT_t *d_colInd_out, const UINT_t num_vertices, const UINT_t num_edges, bool use_double_buffer) {
	std::uint8_t* d_temp_storage{};
	std::size_t temp_storage_bytes{};

	if (use_double_buffer) {
		cub::DoubleBuffer<UINT_t> d_keys(d_colInd_in, d_colInd_out);
		cub::DeviceSegmentedSort::SortKeys(d_temp_storage, temp_storage_bytes, d_keys, num_edges, num_vertices, d_rowPtr, d_rowPtr + 1);
		checkCudaErrors(cudaMalloc((void **)&d_temp_storage, temp_storage_bytes * sizeof(std::uint8_t)));
		cub::DeviceSegmentedSort::SortKeys(d_temp_storage, temp_storage_bytes, d_keys, num_edges, num_vertices, d_rowPtr, d_rowPtr + 1);
		checkCudaErrors(cudaFree(d_temp_storage));
		return d_keys.Current();
	} else {
		cub::DeviceSegmentedSort::SortKeys(d_temp_storage, temp_storage_bytes, d_colInd_in, d_colInd_out, num_edges, num_vertices, d_rowPtr, d_rowPtr + 1);
		checkCudaErrors(cudaMalloc((void **)&d_temp_storage, temp_storage_bytes * sizeof(std::uint8_t)));
		cub::DeviceSegmentedSort::SortKeys(d_temp_storage, temp_storage_bytes, d_colInd_in, d_colInd_out, num_edges, num_vertices, d_rowPtr, d_rowPtr + 1);
		checkCudaErrors(cudaFree(d_temp_storage));
		return d_colInd_out;
	}
}


BIG_GRAPH_TYPE *read_graph(char *filename, bool matrix_market, bool zero_indexed, preprocess_t preprocess_style) {
	FILE *infile = fopen(filename, "r");
	if (infile == NULL) {
		fprintf(stderr, "ERROR: unable to open graph file.\n");
		usage();
	}

	BIG_GRAPH_TYPE *graph = (BIG_GRAPH_TYPE *) malloc(sizeof(BIG_GRAPH_TYPE));
	char line[256];

	/* Skip any header lines */
	do {
		if (fgets(line, sizeof(line), infile) == NULL) usage();
	} while (line[0] < '0' || line[0] > '9');

	/* Skip line if the file is in Matrix Market format. We do not use the given vertex/edge counts. */
	if (matrix_market) {
		if (fgets(line, sizeof(line), infile) == NULL) usage();
	}

	UINT_t vertex_count = 0;
	ULONG_t edge_count = 0;
	size_t size = 10240;
	edge_t* edges = (edge_t*) malloc(size * sizeof(edge_t));
	assert_malloc(edges);

	UINT_t max_vertex = 0;
	UINT_t v, w;

	if (sscanf(line, "%d %d\n", &v, &w) == 2) {
		do {
			if (edge_count >= size) {
				size += 10240;	
				edge_t *new_edges = (edge_t*) realloc(edges, size * sizeof(edge_t));
				assert_malloc(new_edges);
				edges = new_edges;
			}

			if ((!zero_indexed) && (v == 0 || w == 0)) {
				fprintf(stderr, "ERROR: zero vertex id detected but -z was not set.\n");
				usage();
			}

			v -= (zero_indexed ? 0 : 1);
			w -= (zero_indexed ? 0 : 1);

			/* Remove self-loops. */
			if (v != w) {
				max_vertex = max2(max_vertex, max2(v, w));

				/* v->w */
				edges[edge_count].src = v;
				edges[edge_count].dst = w;
				edge_count++;
				/* w->v */
				edges[edge_count].src = w;
				edges[edge_count].dst = v;
				edge_count++;
			}
		} while (fscanf(infile, "%d %d\n", &v, &w) == 2);
	}

	fclose(infile);

	vertex_count = max_vertex + 1;

	/* Sort edges (in order to remove duplicates). */
	if (preprocess_style != PREPROCESS_CPU) {
		edge_t *d_edges;
		edge_t *d_edges_alt;
		edge_t *d_out;

		checkCudaErrors(cudaMalloc((void **)&d_edges, edge_count * sizeof(edge_t)));
		checkCudaErrors(cudaMalloc((void **)&d_edges_alt, edge_count * sizeof(edge_t)));
		checkCudaErrors(cudaMemcpy(d_edges, edges, edge_count * sizeof(edge_t), cudaMemcpyHostToDevice));

		if (preprocess_style == PREPROCESS_GPU_CONSTRAINED)
			d_out = sort_edges_GPU(d_edges, d_edges_alt, edge_count, true);
		else
			d_out = sort_edges_GPU(d_edges, d_edges_alt, edge_count, false);

		checkCudaErrors(cudaMemcpy(edges, d_out, edge_count * sizeof(edge_t), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaFree(d_edges));
		checkCudaErrors(cudaFree(d_edges_alt));
	} else {
		qsort(edges, edge_count, sizeof(edge_t), compareEdge_t);
	}	
	
	ULONG_t *rowPtr = (ULONG_t *) calloc(vertex_count+1, sizeof(ULONG_t));
	assert_malloc(rowPtr);

	ULONG_t edge_count_no_dup = 1;

	edge_t lastedge;
	lastedge.src = edges[0].src;
	lastedge.dst = edges[0].dst;

	UINT_t *colInd = (UINT_t *) edges; /* colInd overwrites the edges array. Possible because sizeof(edge_t) > sizeof(UINT_t). */
	colInd[0] = lastedge.dst;
	rowPtr[lastedge.src + 1]++;

	/* Remove duplicate edges. */
	for (ULONG_t i=1; i<edge_count; i++) {
		if (compareEdge_t(&lastedge, &edges[i]) != 0) {
			colInd[edge_count_no_dup++] = edges[i].dst;
			rowPtr[edges[i].src + 1]++;
			lastedge.src = edges[i].src;
			lastedge.dst = edges[i].dst;
		}
	}

	/* Free excess memory from the colInd/edges array. */
	UINT_t *new_colInd = (UINT_t *) realloc(colInd, edge_count_no_dup * sizeof(UINT_t));

	for (UINT_t v=1; v<=vertex_count; v++) {
		rowPtr[v] += rowPtr[v-1];
	}

	graph->numVertices = vertex_count;
	graph->numEdges = edge_count_no_dup;
	graph->rowPtr = rowPtr;
	graph->colInd = new_colInd;

	return graph;
}

BIG_GRAPH_TYPE *preprocess(const BIG_GRAPH_TYPE *original_graph, preprocess_t preprocess_style) {
	preprocess_vertex_t *vertices = (preprocess_vertex_t *) malloc(original_graph->numVertices * sizeof(preprocess_vertex_t));
	assert_malloc(vertices);

	for (UINT_t v=0; v<original_graph->numVertices; v++) {
		vertices[v].id = v;
		vertices[v].edges = &original_graph->colInd[original_graph->rowPtr[v]];
		vertices[v].num_edges = original_graph->rowPtr[v+1] - original_graph->rowPtr[v];
	}

	if (preprocess_style != PREPROCESS_CPU) {
		preprocess_vertex_t *d_vertices;
		preprocess_vertex_t *d_vertices_alt;
		preprocess_vertex_t *d_out;

		checkCudaErrors(cudaMalloc((void **)&d_vertices, original_graph->numVertices * sizeof(preprocess_vertex_t)));
		checkCudaErrors(cudaMalloc((void **)&d_vertices_alt, original_graph->numVertices * sizeof(preprocess_vertex_t)));
		checkCudaErrors(cudaMemcpy(d_vertices, vertices, original_graph->numVertices * sizeof(preprocess_vertex_t), cudaMemcpyHostToDevice));

		if (preprocess_style == PREPROCESS_GPU_CONSTRAINED)
			d_out = sort_vertices_GPU(d_vertices, d_vertices_alt, original_graph->numVertices, true);
		else
			d_out = sort_vertices_GPU(d_vertices, d_vertices_alt, original_graph->numVertices, false);

		checkCudaErrors(cudaMemcpy(vertices, d_out, original_graph->numVertices * sizeof(preprocess_vertex_t), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaFree(d_vertices));
		checkCudaErrors(cudaFree(d_vertices_alt));
	} else {
		qsort(vertices, original_graph->numVertices, sizeof(preprocess_vertex_t), compare_vertex_degree_ascending);
	}

	UINT_t *reverse = (UINT_t *) malloc(original_graph->numVertices * sizeof(UINT_t));
	assert_malloc(reverse);

	for (UINT_t v=0; v<original_graph->numVertices; v++) {
		reverse[vertices[v].id] = v;
	}

	BIG_GRAPH_TYPE *graph = (BIG_GRAPH_TYPE *) malloc(sizeof(BIG_GRAPH_TYPE));
	assert_malloc(graph);

	graph->numVertices = original_graph->numVertices;
	graph->numEdges = original_graph->numEdges/2;

	graph->rowPtr = (ULONG_t*) malloc((graph->numVertices + 1) * sizeof(ULONG_t));
	assert_malloc(graph->rowPtr);
	graph->colInd = (UINT_t*) malloc(graph->numEdges * sizeof(UINT_t));
	assert_malloc(graph->colInd);

	ULONG_t edge_count = 0;

	graph->rowPtr[0] = 0;

	for (UINT_t v=0; v<original_graph->numVertices; v++) {
		UINT_t new_degree = 0;

		for (UINT_t j=0; j<vertices[v].num_edges; j++) {
			UINT_t w = vertices[v].edges[j];
			UINT_t w_new = reverse[w];

			if (w_new > v) {
				graph->colInd[edge_count++] = w_new;
				new_degree++;
			}
		}

		graph->rowPtr[v+1] = graph->rowPtr[v] + new_degree;

		if (preprocess_style == PREPROCESS_CPU) {
			qsort(&graph->colInd[graph->rowPtr[v]], new_degree, sizeof(UINT_t), compareInt_t);
		}
	}

	free(vertices);
	free(reverse);

	if (preprocess_style != PREPROCESS_CPU) {
		ULONG_t *d_rowPtr;
		UINT_t *d_colInd;
		UINT_t *d_colInd_alt;
		UINT_t *d_colInd_out;

		checkCudaErrors(cudaMalloc((void **)&d_rowPtr, (graph->numVertices+1) * sizeof(ULONG_t)));
		checkCudaErrors(cudaMalloc((void **)&d_colInd, graph->numEdges * sizeof(UINT_t)));
		checkCudaErrors(cudaMalloc((void **)&d_colInd_alt, graph->numEdges * sizeof(UINT_t)));
		checkCudaErrors(cudaMemcpy(d_rowPtr, graph->rowPtr, (graph->numVertices+1) * sizeof(ULONG_t), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_colInd, graph->colInd, graph->numEdges * sizeof(UINT_t), cudaMemcpyHostToDevice));

		if (preprocess_style == PREPROCESS_GPU_CONSTRAINED)
			d_colInd_out = sort_colInd_GPU(d_rowPtr, d_colInd, d_colInd_alt, graph->numVertices, graph->numEdges, true);
		else
			d_colInd_out = sort_colInd_GPU(d_rowPtr, d_colInd, d_colInd_alt, graph->numVertices, graph->numEdges, false);

		checkCudaErrors(cudaMemcpy(graph->colInd, d_colInd_out, graph->numEdges * sizeof(UINT_t), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaFree(d_rowPtr));
		checkCudaErrors(cudaFree(d_colInd));
		checkCudaErrors(cudaFree(d_colInd_alt));
	}

	return graph;
}

void free_graph(BIG_GRAPH_TYPE *graph) {
	free(graph->rowPtr);
	free(graph->colInd);
	free(graph);
}

int main(int argc, char **argv) {
	char *graph_filename = NULL;
	bool graph_mm = false;
	bool graph_zero_indexed = false;
	UINT_t loop_cnt = 1;

	/* Default: use lightweight GPU-based preprocessing (worst case ~ m*8 device memory). */
	preprocess_t preprocess_style = PREPROCESS_GPU_CONSTRAINED;

	while ((argc > 1) && (argv[1][0] == '-')) {
		switch (argv[1][1]) {
			case 'm':
				graph_mm = true;
			case 'e':
				if (argc < 3) usage();
				graph_filename = argv[2];
				if (graph_filename == NULL) usage();
				argv+=2;
				argc-=2;
				break;
			case 'z':
				graph_zero_indexed = true;
				argv++;
				argc--;
				break;
			case 'l':
				if (argc < 3) usage();
				loop_cnt = atoi(argv[2]);
				argv+=2;
				argc-=2;
				break;
			case 'p':
				if (argc < 3) usage();
				if (atoi(argv[2]) < PREPROCESS_CPU || atoi(argv[2]) > PREPROCESS_GPU_CONSTRAINED) usage();
				preprocess_style = (preprocess_t) atoi(argv[2]);
				argv+=2;
				argc-=2;
				break;
		}
	}

	if (graph_filename == NULL) usage();

	BIG_GRAPH_TYPE *original_graph = read_graph(graph_filename, graph_mm, graph_zero_indexed, preprocess_style);
	BIG_GRAPH_TYPE *graph = preprocess(original_graph, preprocess_style);
	free_graph(original_graph);

	printf("%-60s %16s %22s %24s %16s %16s %16s %16s\n",
		"graph", "n", "m", "retrieval", "GPU copy (s)", "GPU exec (s)", "GPU total (s)", "CPU+GPU (s)");

	const char *strats_names[4] = {"edge_list", "edge_list_src", "binary_search", "binary_search_cached"};
	void (*(strats_functions[4]))(const BIG_GRAPH_TYPE *, GPU_time *) = {edge_get_edgelist_GPU, edge_get_edgelist_src_only_GPU, edge_get_binary_search_GPU, edge_get_binary_search_cached_GPU};

	for (UINT_t strat=0; strat<4; strat++) {
		bool warmed_up = false;

		for (UINT_t i=0; i<(loop_cnt+1); i++) {
			double t_cpu = get_seconds();
			GPU_time t_gpu = { .copy=0.0, .exec=0.0 };

			strats_functions[strat](graph, &t_gpu);

			t_cpu = get_seconds() - t_cpu;

			t_gpu.copy /= (double) 1000;
			t_gpu.exec /= (double) 1000;

			if (warmed_up) {
				printf("%-60s %16d %22lu %24s %16.6f %16.6f %16.6f %16.6f\n",
					graph_filename, graph->numVertices, graph->numEdges, strats_names[strat], t_gpu.copy, t_gpu.exec, t_gpu.copy + t_gpu.exec, t_cpu);
			} else {
				warmed_up = true;
			}
		}
	}

	free_graph(graph);

	return EXIT_SUCCESS;
}