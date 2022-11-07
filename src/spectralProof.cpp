#include <iostream>

#include <pcl/io/ply_io.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/kdtree/kdtree_flann.h>

#include <armadillo>

void VisualizeCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
{
    pcl::visualization::CloudViewer viewer("Cloud Viewer");
    viewer.showCloud(cloud);
    std::cout << '\n' << "Press Enter";
    while (std::cin.get() != '\n') {}
}

void ReadPointCloud(std::string &ply_file, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
{
    pcl::PCLPointCloud2::Ptr cloud2(new pcl::PCLPointCloud2 ());

    pcl::io::loadPLYFile(ply_file, *cloud2);
    pcl::fromPCLPointCloud2(*cloud2, *cloud);
}

double FindMinRadius(pcl::KdTreeFLANN<pcl::PointXYZ> &kdTree, int size)
{
    double radius = 0.1f;
    for (int i = 0; i < size; i++) {
        std::vector<int> indicies_found;
        std::vector<float> squaredDistances;
        kdTree.radiusSearch(i, radius, indicies_found, squaredDistances, 100);

        int num_edges = indicies_found.size() - 1;
        // If the minimum edges end up equaling 1 then it is a disconnected graph.
        // reset the laplacian, reset the counter and increase the radius until we
        // have a fully connected graph
        if (num_edges == 0)
        {
            std::cout << "Increasing radius" << std::endl;
            i = 0;
            radius += 0.1f;
        }
    }

    return radius;
}

double GetSmallestDistance(pcl::KdTreeFLANN<pcl::PointXYZ> &kdTree, int size)
{
    double min = 1000000.0;
    for ( int i = 0; i < size; i++ )
    {
        std::vector<int> indicies_found;
        std::vector<float> squaredDistances;
        kdTree.nearestKSearch(i, 2, indicies_found, squaredDistances);
        min = ( squaredDistances[1] < min ) ? sqrt(squaredDistances[1]) : min;
    }

    return min;
}

void PopulateLaplacian(pcl::KdTreeFLANN<pcl::PointXYZ> &kdTree, arma::sp_mat &laplacian, int size, double radius)
{
     double smallestDistance = GetSmallestDistance(kdTree, size);
     double bias = 1.0 - smallestDistance;

    // TODO change max_nn to max points in the cloud to test later
    unsigned int max_nn = 1000;

    std::cout << "Laplacian size: " << size << std::endl;
    for (int i = 0; i < size; i++) {
        std::vector<int> indicies_found;
        std::vector<float> squaredDistances;
        kdTree.radiusSearch(i, radius, indicies_found, squaredDistances, max_nn);

        int num_edges = indicies_found.size() - 1;
        laplacian(i,i) = num_edges;

        for (int j = 1; j < indicies_found.size(); j++)
        {
            laplacian(i, indicies_found[j]) = -1 / (sqrt(squaredDistances[j]) + bias);
            laplacian(indicies_found[j], i) = -1 / (sqrt(squaredDistances[j]) + bias);
        }
    }
}

void ComputeEigensSparse(arma::sp_mat laplacian, arma::vec eigval, arma::mat eigvec)
{
    auto start = std::chrono::steady_clock::now();
    arma::eigs_sym(eigval, eigvec, laplacian, laplacian.n_cols - 1);
    auto end = std::chrono::steady_clock::now();
    cout << "Elapsed time in miliseconds: "
        << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
        << " ms" << endl;
}

void ComputeLaplacian(arma::sp_mat &laplacian, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
{
    pcl::KdTreeFLANN<pcl::PointXYZ> kdTree;
    kdTree.setInputCloud(cloud);

    int rows = cloud->points.size();
    int cols = rows;

    laplacian = arma::sp_mat(rows, cols);

    double radius = FindMinRadius(kdTree, rows);

    PopulateLaplacian(kdTree, laplacian, rows, radius);
}

int main()
{
    std::string teddyBearFile = "/home/nate/Datasets/teddyPly/b7.ply";
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    arma::sp_mat laplacian;
    arma::vec eigval;
    arma::mat eigvec;

    ReadPointCloud(teddyBearFile, cloud);
    ComputeLaplacian(laplacian, cloud);
    ComputeEigensSparse(laplacian, eigval, eigvec);
    // Plt eigenvalues in histogram
    // plt eigenvectors in histogram
    std::cout << "Hello World" << std::endl;
    return 1;
}
