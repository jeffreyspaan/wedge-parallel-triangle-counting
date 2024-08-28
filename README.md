# Wedge-Parallel Triangle Counting
See [...]

## Requirements

CUDA (tested on 12.4), gcc

## Compile & run

To compile the code use (and match your target architecture in the `Makefile`, the default is `sm_86`):

    $ make

This creates one executable named `tc` which includes both the preprocessing and the GPU code.

To clean use:

    $ make clean

To run the code:

    $ ./tc -m <Matrix Market graph> -s <spread> -a <adjacency matrix length>

or: 

    $ ./tc -e <edge list graph> -s <spread> -a <adjacency matrix length>

Run `./tc` for more usage information.

## Example

    $ ./tc -m Amazon0302.mtx -s 5 -a 8192 -l 10

gives the following output:

    $ graph                                                                       n                m                s                a        triangles       prepro (s)     GPU copy (s)     GPU exec (s)    GPU total (s)      CPU+GPU (s)
    $ Amazon0302.mtx                                                         262111           899792                5             8192           717719         0.056998         0.002199         0.000407         0.002606         0.003735
    $ Amazon0302.mtx                                                         262111           899792                5             8192           717719         0.056998         0.002165         0.000406         0.002570         0.004149
    $ Amazon0302.mtx                                                         262111           899792                5             8192           717719         0.056998         0.002393         0.000438         0.002831         0.004080
    $ ...

## Reference

[...]
