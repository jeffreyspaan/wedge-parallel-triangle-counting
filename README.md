# WeTriC: Wedge-Parallel Triangle Counting for GPUs
This repository contains the source code for the WeTriC algorithm from the paper "Wedge-Parallel Triangle Counting for GPUs". For the complete artifact, visit the [Zenodo](https://doi.org/10.5281/zenodo.15611507) repository.

## Requirements

CUDA (tested on 12.3), cub (part of [CCCL](https://github.com/nvidia/cccl)) (included in the CUDA Toolkit), a c++ compiler.

## Compile & run

To compile the code use (and match your target architecture in the `Makefile`, the default is `sm_86`):

    $ make

This creates one executable named `tc` which includes both the preprocessing (reading the graph, reodering, etc.) and the GPU code.

To clean use:

    $ make clean

To run the code:

    $ ./tc -e <edge list graph> -s <spread> -a <adjacency matrix length>

or: 

    $ ./tc -m <Matrix Market graph> -s <spread> -a <adjacency matrix length>

Run `./tc` for more usage information.

## Example

    $ ./tc -m Amazon0302.mtx -s 5 -a 8192 -l 10

gives the following output:

    $ graph                                                                       n                m                s                a        triangles       prepro (s)     GPU copy (s)     GPU exec (s)    GPU total (s)      CPU+GPU (s)
    $ Amazon0302.mtx                                                         262111           899792                5             8192           717719         0.056998         0.002199         0.000407         0.002606         0.003735
    $ Amazon0302.mtx                                                         262111           899792                5             8192           717719         0.056998         0.002165         0.000406         0.002570         0.004149
    $ Amazon0302.mtx                                                         262111           899792                5             8192           717719         0.056998         0.002393         0.000438         0.002831         0.004080
    $ ...

## Experiments

The experiments folder contains all implementations of all additional algorithms used in the paper. This includes:

- implementations for different edge-retrieval strategies.
- an implementation for a wedge-parallel arrow-wedge-style algorithm.
- implementations for the vertex- and edge-parallel algorithms (with outgoing, arrow, and mixed wedge styles).
- implementations for WeTriC without optimizations, WeTriC with reordering, and WeTriC with reordering and spreading (note: for WeTriC with reordering, spreading, and cooperation use the main implementation with `-a` set to `0`).
- implementations for versions of most of the above algorithms able to handle very large graphs (m > 2^32), see the `tc_big_graphs...` files.

## Reference

[...]
