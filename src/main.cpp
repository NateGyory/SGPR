#include <iostream>
#include <thread>
#include <chrono>

#include <pcl/io/ply_io.h>
#include <pcl/conversions.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/kdtree/kdtree_flann.h>

#include <Eigen/Core>
//#include <Eigen/Eigenvalues>

using namespace std::chrono_literals;

int main()
{
    std::string handFile = "/home/nate/Datasets/handsPly/b1.ply";
    pcl::PCLPointCloud2::Ptr cloud2(new pcl::PCLPointCloud2 ());
    pcl::PCLPointCloud2::Ptr cloud2_filtered(new pcl::PCLPointCloud2 ());
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::io::loadPLYFile(handFile, *cloud2);


    pcl::VoxelGrid<pcl::PCLPointCloud2> voxGrid;
    voxGrid.setInputCloud(cloud2);
    voxGrid.setLeafSize (6.0f, 6.0f, 6.0f);
    voxGrid.filter(*cloud2_filtered);

    pcl::fromPCLPointCloud2(*cloud2_filtered, *cloud);

    pcl::KdTreeFLANN<pcl::PointXYZ> kdTree;
    kdTree.setInputCloud(cloud);
    pcl::PointXYZ searchPoint;

    std::vector<int> indicies_found;
    std::vector<float> radiusSquaredDistance;
    int max_nn = 100;

    float radius = 6.0f;

    int rows = cloud->size();
    int cols = rows;

    std::cout << "Data size: " << rows << std::endl;

    Eigen::MatrixXd laplacian = Eigen::MatrixXd::Zero(rows, cols);
    Eigen::MatrixXd laplacian2 = Eigen::MatrixXd::Zero(rows, cols);

    std::cout << "Populating Laplacian" << std::endl;
    for (int i = 0; i < rows; i++)
    {
        kdTree.radiusSearch(i, radius, indicies_found, radiusSquaredDistance, max_nn);

        // Populate the laplacian diaganol w/ edge count
        int num_edges = indicies_found.size();
        laplacian(i,i) = num_edges;

        // Populate the adjacency Matrix with -1 where edge is found
        // TODO: probably will make a weighted adjacency solution with distances
        for (int j = 0; j < indicies_found.size(); j++)
        {
            laplacian(i, indicies_found[j]) = 1 * radiusSquaredDistance[j];
            // Since laplcian is symmetric populare the j, i as well
            laplacian(indicies_found[j], i) = 1 * radiusSquaredDistance[j];
        }
    }

    std::cout << "Populating Laplacian" << std::endl;
    for (int i = 0; i < rows; i++)
    {
        kdTree.radiusSearch(i, radius, indicies_found, radiusSquaredDistance, max_nn);

        // Populate the laplacian diaganol w/ edge count
        int num_edges = indicies_found.size();
        int new_idx = rows - 1 - i;
        laplacian2(new_idx, new_idx) = num_edges;

        // Populate the adjacency Matrix with -1 where edge is found
        // TODO: probably will make a weighted adjacency solution with distances
        for (int j = 0; j < indicies_found.size(); j++)
        {
            int new_idx2 = rows - 1 - indicies_found[j];
            laplacian2(new_idx, new_idx2) = 1 * radiusSquaredDistance[j];
            // Since laplcian is symmetric populare the j, i as well
            laplacian2(new_idx2, new_idx) = 1 * radiusSquaredDistance[j];
        }
    }

    std::cout << "Solving eigenvalues" << std::endl;
    // Compute eigenvalues of the Laplacian matrix
    Eigen::EigenSolver<Eigen::MatrixXd> eigenSolver(laplacian);
    std::cout << "Solving eigenvalues 2" << std::endl;
    Eigen::EigenSolver<Eigen::MatrixXd> eigenSolver2(laplacian2);
    std::cout << eigenSolver.eigenvalues() << std::endl;
    std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
    std::cout << eigenSolver2.eigenvalues() << std::endl;

    //pcl::visualization::CloudViewer viewer("Cloud Viewer");
    //viewer.showCloud(cloud);
    //while(1);

    return 1;
}
