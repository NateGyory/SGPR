#include <iostream>
#include <thread>
#include <chrono>

#include <pcl/io/ply_io.h>
#include <pcl/conversions.h>
#include <pcl/visualization/cloud_viewer.h>

using namespace std::chrono_literals;

int main()
{
    // TODO:
    // 1) Get the CloudViewer to work
    // 2) Apply VoxelGrid filter
    std::string handFile = "/home/nate/Datasets/teddyPly/test.ply";
    pcl::PCLPointCloud2 cloud2;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::io::loadPLYFile(handFile, cloud2);

    pcl::fromPCLPointCloud2( cloud2, *cloud);


    pcl::visualization::CloudViewer viewer("Cloud Viewer");
    viewer.showCloud(cloud);
}
