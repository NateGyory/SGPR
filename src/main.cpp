#include<iostream>
#include "pcl/io/ply_io.h"

int main()
{
    // TODO:
    // 1) Add Open3D
    // 2) Render hand in Open3D
    // 3) Apply VoxelGrid filter to hand
    std::string handFile = "/home/nate/Datasets/handsPly/b1.ply";
    pcl::PCLPointCloud2 cloud;
    pcl::io::loadPLYFile(handFile, cloud);

    std::cout << cloud << std::endl;
}
