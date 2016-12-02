//#include <helper_cuda.h>
#include "kernel.cuh"

__global__ void my_first_kernel(float *x)
{
    int tid = threadIdx.x + blockDim.x*blockIdx.x;
    x[tid] = (float) 2.0f;//threadIdx.x;
}

void my_first(float *x, int nblocks, int nthreads)
{
    my_first_kernel<<<nblocks,nthreads>>>(x);
}
