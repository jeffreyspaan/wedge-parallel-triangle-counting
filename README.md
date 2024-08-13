# Wedge-Parallel Triangle Counting
See [...]

## Requirements

>CUDA 12.4**, gcc

## Compile & run

To compile the code use (and match your target architecture in the `Makefile`, the default is `-arch native`):

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

    $ ./tc -m Amazon0302.mtx -s 5 -a 8192

gives the following output:

    $ graph                                                                       n                m                s                a        triangles       prepro (s)     GPU copy (s)     GPU exec (s)    GPU total (s)      CPU+GPU (s)
    $ Amazon0302.mtx                                                         262111           899792                5             8192           717719         0.055316         0.002182         0.000406         0.002587         0.003893
    $ Amazon0302.mtx                                                         262111           899792                5             8192           717719         0.055316         0.002021         0.000409         0.002430         0.003575
    $ Amazon0302.mtx                                                         262111           899792                5             8192           717719         0.055316         0.002194         0.000407         0.002601         0.003834
    $ ...

## Reference

[...]
