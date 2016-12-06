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

    Eigen::Matrix<float, 3, 5> extremas = Eigen::Matrix<float, 3, 5>::Zero();
    extremas(0, 1) = 1.0f; extremas(0, 2) = -1.0f; extremas(1, 3) = 1.0f; extremas(1, 4) = -1.0f;

    cv::Mat image = cv::Mat::zeros(480, 640, CV_8UC4);
    cv::Mat depth = cv::Mat::zeros(480, 640, CV_16UC1);
    int counter = 0;

    auto start = std::chrono::system_clock::now();
    for (SurfelT& s : cloud->points) {

        if (s.z < 0) {
            ++counter;
            continue;
        }

        Eigen::Vector3f p = K*s.getVector3fMap();
        p = 1.0f/p(2)*p;
        if (p(0) < 0 || p(0) > 640 || p(1) < 0 || p(1) > 480) {
            ++counter;
            continue;
        }

        Eigen::Matrix<float, 3, 5> rextremas = K*((s.radius*extremas).colwise() + s.getVector3fMap());
        rextremas = rextremas.array().rowwise()/rextremas.row(2).array();
        int minx = std::max(int(rextremas.row(0).minCoeff()), 0);
        int maxx = std::min(int(rextremas.row(0).maxCoeff()), 640);
        int miny = std::max(int(rextremas.row(1).minCoeff()), 0);
        int maxy = std::min(int(rextremas.row(1).maxCoeff()), 480);

        Eigen::Vector3f nx = s.getNormalVector3fMap();
        nx.normalize();
        float r2 = 1.0/(s.radius*s.radius);
        Eigen::Matrix3f NN = nx*nx.transpose();
        Eigen::Matrix3f A = eps2*NN + r2*(Eigen::Matrix3f::Identity() - NN);
        Eigen::Vector3f c = s.getVector3fMap();

        Eigen::Matrix3f AA = Kinv.transpose()*((c.transpose()*A*c - 1.0f)*A - A*c*c.transpose()*A)*Kinv;
        Eigen::Matrix2f A2 = AA.block<2, 2>(0, 0);
        Eigen::Vector2f B2 = 2.0f*AA.block<2, 1>(0, 2);
        float C2 = AA(2, 2);

        for (int i = miny; i < maxy; ++i) {

            for (int j = minx; j < maxx; ++j) {
                Eigen::Vector2f v;
                v << float(j), float(i);
                if (C2 + B2.transpose()*v + v.transpose()*A2*v <= 0.0f) {
                    uint16_t dval = uint16_t(5000*s.z);
                    uint16_t& prev = depth.at<uint16_t>(i, j);
                    if (prev == 0 || prev > dval) {
                        image.at<uint32_t>(i, j) = s.rgba;
                        depth.at<uint16_t>(i, j) = dval;
                    }
                }
            }
        }

        ++counter;
    }

    auto duration = std::chrono::duration_cast<TimeT>(std::chrono::system_clock::now() - start);

    cout << "Elapsed duration " << duration.count() << " ms" << endl;

    cv::imshow("image", image);
    cv::imshow("depth", depth);
    cv::waitKey();
    return 0;
}
