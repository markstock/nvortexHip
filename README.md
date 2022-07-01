# nvortexHip
Simple direct solver for vortex particle methods using HIP for GPU

## Compile and run
As long CUDA is installed and nvcc is in your PATH, you should be able to do:

    make
    ./nvHip05.bin -n=100000

## Hipify
The first four of these codes were created from [nvortexCuda](https://github.com/markstock/nvortexCuda) using

    hipify-clang nvCuda04.cu -o=nvHip04.hip -v --cuda-path=[..] -I [..]/include


