#ifndef PROJECT_KERNEL_CUH
#define PROJECT_KERNEL_CUH

#include <cuda.h>

void project(float *d_x, float *d_y, float *d_z, float *d_nx,
             float *d_ny, float *d_nz, float *d_r, float *d_rgba,
             int nblocks, int nthreads, float neps2, float* nK, float* nKinv, float* himage);

#endif // PROJECT_KERNEL_CUH
