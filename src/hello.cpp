#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include "cuda/kernel.cuh"

int main(int argc, char **argv) {
    float *h_x, *d_x; // h=host, d=device
    int nblocks=2, nthreads=8, nsize=2*8;
    h_x = (float *)malloc(nsize*sizeof(float));
    cudaMalloc((void **)&d_x,nsize*sizeof(float));

    my_first(d_x,nblocks,nthreads);

    cudaMemcpy(h_x,d_x,nsize*sizeof(float),cudaMemcpyDeviceToHost);
    for (int n=0; n<nsize; n++)
        printf(" n,x = %d %f \n",n,h_x[n]);
    cudaFree(d_x);
    free(h_x);
}
