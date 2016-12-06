//#include <helper_cuda.h>
#include "project_kernel.cuh"
#include <stdio.h>

__constant__ float K[3][3];
__constant__ float Kinv[3][3];
__constant__ float eps2;
__device__  float image[480*640];

__global__ void project_kernel(float *d_x, float *d_y, float *d_z, float *d_nx,
                               float *d_ny, float *d_nz, float *d_r, float *d_rgba)
{
    // each kernel function gets a surfel in the cloud

    // assume global K, assume points have already been transformed into camera frame?

    // now, based on normal and radius determine the pixels covered
    // we could probably use the projection matrix on the ellipse somehow
    //int tid = threadIdx.x + blockDim.x*blockIdx.x;
    int i = threadIdx.x + blockDim.x*blockIdx.x;

    if (d_z[i] < 0.0f) {
        printf("Index: %d, Depth: %f\n", i, d_z[i]);
        printf("K: [%f, %f, %f; %f, %f, %f; %f, %f, %f]\n", K[0][0], K[0][1], K[0][2], K[1][0], K[1][1], K[1][2], K[2][0], K[2][1], K[2][2]);
        return;
    }

    float px = K[0][0]*d_x[i]/d_z[i] + K[0][2];
    float py = K[2][2]*d_y[i]/d_z[i] + K[1][2];
    if (px < 0 || px > 640 || py < 0 || py > 480) {
        return;
    }

    float r2 = 1.0f/(d_r[i]*d_r[i]);

    float NN[3][3];
    NN[0][0] = d_nx[i]*d_nx[i]; NN[0][1] = d_nx[i]*d_ny[i]; NN[0][2] = d_nx[i]*d_nz[i];
    NN[1][0] = d_ny[i]*d_nx[i]; NN[1][1] = d_ny[i]*d_ny[i]; NN[1][2] = d_ny[i]*d_nz[i];
    NN[2][0] = d_nz[i]*d_nx[i]; NN[2][1] = d_nz[i]*d_ny[i]; NN[2][2] = d_nz[i]*d_nz[i];

    float A[3][3];
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            A[row][col] = (r2 - eps2)*NN[row][col];
        }
        A[row][row] += eps2;
    }

    float c[3] = {d_x[i], d_y[0], d_z[0]};

    float Ac[3];
    for (int row = 0; row < 3; ++row) {
        Ac[row] = 0.0f;
        for (int col = 0; col < 3; ++col) {
            Ac[row] += A[row][col]*c[col];
        }
    }

    float cAc = 0.0f;
    for (int row = 0; row < 3; ++row) {
        cAc += c[row]*Ac[row];
    }

    // now we reuse A to compute AA
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            A[row][col] = (cAc - 1.0f)*A[row][col] - Ac[row]*Ac[col];
        }
    }

    // and finally, let's reuse NN and A to compute the AA with camera matrix
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            // A*Kinv
            NN[row][col] = A[row][0]*Kinv[0][col] + A[row][1]*Kinv[1][col] + A[row][2]*Kinv[2][col];
        }
    }

    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            // this time Kinv^T*A
            A[row][col] = Kinv[0][row]*NN[0][col] + Kinv[1][row]*NN[1][col] + Kinv[2][row]*NN[2][col];
        }
    }

    // now, let's get the camera matrix in there


    //Eigen::Vector3f c = s.getVector3fMap();

//    Eigen::Matrix3f AA = Kinv.transpose()*((c.transpose()*A*c - 1.0f)*A - A*c*c.transpose()*A)*Kinv;
//    Eigen::Matrix2f A2 = AA.block<2, 2>(0, 0);
//    Eigen::Vector2f B2 = 2.0f*AA.block<2, 1>(0, 2);
//    float C2 = AA(2, 2);
    int minx = max(int(px) - 5, 0); int maxx = min(int(px) + 5, 640); int miny = max(int(py) - 5, 0); int maxy = min(int(py) + 5, 480);

    float x, y, disc;
    for (int row = miny; row < maxy; ++row) {

        for (int col = minx; col < maxx; ++col) {
            x = col; y = row;
            // C2 + B2.transpose()*v + v.transpose()*A2*v
            disc = A[0][0] + 2.0f*(A[0][2]*x + A[1][2]*y) + A[0][0]*x*x + 2.0f*A[0][1]*x*y + A[1][1]*y*y;
            if (disc < 0.0f) {
                x = 1.0f;
                image[640*row+col] = d_rgba[i];
            }
        }
    }

}

void project(float *d_x, float *d_y, float *d_z, float *d_nx,
             float *d_ny, float *d_nz, float *d_r, float *d_rgba,
             int nblocks, int nthreads, float neps2, float* nK, float* nKinv, float* himage)
{
    cudaMemcpyToSymbol(K, nK, 9*sizeof(float));
    cudaMemcpyToSymbol(Kinv, nKinv, 9*sizeof(float));
    cudaMemcpyToSymbol(&eps2, &neps2, sizeof(float));
    project_kernel<<<nblocks,nthreads>>>(d_x, d_y, d_z, d_nx, d_ny, d_nz, d_r, d_rgba);
    cudaMemcpyFromSymbol(himage, image, 480*640*sizeof(float));
}
