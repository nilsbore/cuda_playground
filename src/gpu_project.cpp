#include <eigen3/Eigen/Dense>
#include <iostream>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <surfel_type.h>
#include <pcl/io/pcd_io.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <chrono>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include "cuda/project_kernel.cuh"

using SurfelT = SurfelType;
using CloudT = pcl::PointCloud<SurfelT>;
using TimeT = std::chrono::milliseconds;

using namespace std;

int main(int argc, char** argv)
{
    if (argc < 2) {
        cout << "Please provide the surfel map to view!" << endl;
    }

    string surfel_path(argv[1]);
    CloudT::Ptr cloud(new CloudT);
    pcl::io::loadPCDFile(surfel_path, *cloud);

    float eps2 = 1.0f/(0.001f*0.001f);

    Eigen::Matrix3f K;
    K << 533.79638671875, 0.0, 314.86334228515625, 0.0, 533.1127319335938, 241.27134704589844, 0.0, 0.0, 1.0;
    Eigen::Matrix3f Kinv = K.inverse();

    cv::Mat image = cv::Mat::zeros(480, 640, CV_8UC4);
    //cv::Mat depth = cv::Mat::zeros(480, 640, CV_16UC1);

    int nsize = cloud->size();

    float *h_x, *h_y, *h_z, *h_nx, *h_ny, *h_nz, *h_r, *h_rgba;
    float *d_x, *d_y, *d_z, *d_nx, *d_ny, *d_nz, *d_r, *d_rgba;

    h_x = (float *)malloc(nsize*sizeof(float));
    h_y = (float *)malloc(nsize*sizeof(float));
    h_z = (float *)malloc(nsize*sizeof(float));
    h_nx = (float *)malloc(nsize*sizeof(float));
    h_ny = (float *)malloc(nsize*sizeof(float));
    h_nz = (float *)malloc(nsize*sizeof(float));
    h_r = (float *)malloc(nsize*sizeof(float));
    h_rgba = (float *)malloc(nsize*sizeof(float));

    size_t i = 0;
    for (const SurfelT& s : cloud->points) {
        h_x[i] = s.x;
        h_y[i] = s.y;
        h_z[i] = s.z;
        h_nx[i] = s.normal_x;
        h_ny[i] = s.normal_y;
        h_nz[i] = s.normal_z;
        h_r[i] = s.radius;
        (uint32_t&)h_rgba[i] = s.rgba;
        ++i;
    }

    cudaMalloc((void **)&d_x,nsize*sizeof(float));
    cudaMalloc((void **)&d_y,nsize*sizeof(float));
    cudaMalloc((void **)&d_z,nsize*sizeof(float));
    cudaMalloc((void **)&d_nx,nsize*sizeof(float));
    cudaMalloc((void **)&d_ny,nsize*sizeof(float));
    cudaMalloc((void **)&d_nz,nsize*sizeof(float));
    cudaMalloc((void **)&d_r,nsize*sizeof(float));
    cudaMalloc((void **)&d_rgba,nsize*sizeof(float));

    cudaMemcpy(d_x,h_x,nsize*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_y,h_y,nsize*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_z,h_z,nsize*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_nx,h_nx,nsize*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_ny,h_ny,nsize*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_nz,h_nz,nsize*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_r,h_r,nsize*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_rgba,h_rgba,nsize*sizeof(float),cudaMemcpyHostToDevice);

    int nthreads=6*32;
    int nblocks = std::ceil(double(nsize) / double(nthreads));

    auto start = std::chrono::system_clock::now();

    K.transposeInPlace();
    Kinv.transposeInPlace();

    cout << K << endl;
    project(d_x, d_y, d_z, d_nx, d_ny, d_nz, d_r, d_rgba, nblocks, nthreads,
            eps2, K.data(), Kinv.data(), (float*)image.data, nsize);
    cudaMemcpy(h_x,d_x,nsize*sizeof(float),cudaMemcpyDeviceToHost);
    auto duration = std::chrono::duration_cast<TimeT>(std::chrono::system_clock::now() - start);

    cout << "Elapsed duration " << duration.count() << " ms" << endl;

    cudaFree(d_x); cudaFree(d_y); cudaFree(d_z); cudaFree(d_nx); cudaFree(d_ny); cudaFree(d_nz); cudaFree(d_r); cudaFree(d_rgba);

    cv::imshow("image", image);
    //cv::imshow("depth", depth);
    cv::waitKey();
    return 0;
}
