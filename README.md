# nvortexHip
Simple direct solver for vortex particle methods using HIP for GPU

## Compile and run
As long Rocm is installed and `hipcc` is in your PATH, you should be able to do:

    make
    ./nvHip05.bin -n=100000

## Hipify
The first four of these codes were created from [nvortexCuda](https://github.com/markstock/nvortexCuda) using:

    hipify-clang nvCuda04.cu -o=nvHip04.hip -v --cuda-path=[..] -I [..]/include

## Description
This repository contains a few progressive examples of a compute-only n-body calculation
of the Biot-Savart influence of N desingularized vorticies on one another.

Program `nvHip01.cu` is the simplest implementation. On the CPU side, the program parallelizes
with a basic OpenMP `parallel for` loop over the target particles. On the GPU side, we use HIP
without unified or pinned memory (full transfers), with one target particle per "thread."

Program `nvHip02.cu` speeds this up considerably. The CPU now uses `omp simd` to vectorize the
inner loop over source particles. The GPU uses shared memory to load blocks of source particles
in a coalesced manner before all threads operate on that block. This program represents the
"80" part of the "80-20 rule": that you can go most of the way with some simple methods.

Program `nvHip03` adds some enhancements in an attempt to eke out even more performance, though
only on the GPU side. First, we moved the GPU timers to not count allocation and deallocation,
in order to be more consistent with the CPU timers. Second, we now break the computation up
along the source-particle dimension, to allow for greater concurrency, which requires `atomicAdd`
to write results back to main GPU memory. Finally, we added support for multiple GPU systems.

`nvHip04` saves one flop per inner loop by presquaring the target radius, but adds 
six more by performing Kahan summation on the accumulators. This further reduces errors inherent
in summing large arrays of numbers, but seems incompatible with the `omp simd` clause.

Finally, `nvHip05` speeds up the CPU calculation by blocking (putting into hierarchical "blocks"),
which improves memory locality, especially for large N. This has the additional benefit of reducing
the errors inherent in summing large sequences of numbers (so we remove Kahan summation).
It also rearranges some of the asynchronous GPU calls to reduce total memory-move-and-execution time
by a few percent for multi-gpu runs.

