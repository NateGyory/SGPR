#include <iostream>

#include <pcl/io/ply_io.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <armadillo>

#include <matplot/matplot.h>

//#include <Eigen/Core>  // If I want to use spectra
//#include <Eigen/SparseCore>
//#include <Spectra/GenEigsSolver.h>
//#include <Spectra/MatOp/SparseGenMatProd.h>
//#include <iostream>

//using namespace Spectra;

using namespace std::chrono_literals;
using namespace matplot;

void VisualizeCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud)
{
    pcl::visualization::CloudViewer viewer("Cloud Viewer");
    viewer.showCloud(cloud);
    std::cout << '\n' << "Press Enter";
    while (std::cin.get() != '\n') {}
}

void FilterCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud)
{
    double leafInc = 0.01;
    double leafValue = 0.1;
    while (cloud->points.size() > 10000)
    {
        std::cout << "Filtering cloud" << std::endl;
        pcl::PointCloud<pcl::PointXYZ>::Ptr vox_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::VoxelGrid<pcl::PointXYZ> sor;
        sor.setInputCloud (cloud);
        sor.setLeafSize (leafValue, leafValue, leafValue);
        sor.filter (*vox_cloud);

        cloud.reset();
        cloud = vox_cloud;
        leafValue += leafInc;
    }
}

void ReadPointCloud(std::string &ply_file, pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud)
{
    pcl::PCLPointCloud2::Ptr cloud2(new pcl::PCLPointCloud2 ());

    pcl::io::loadPLYFile(ply_file, *cloud2);
    pcl::fromPCLPointCloud2(*cloud2, *cloud);

    FilterCloud(cloud);
}

double FindMinRadius(pcl::KdTreeFLANN<pcl::PointXYZ> &kdTree, int size)
{
    double radius = 0.01f;
    for (int i = 0; i < size; i++) {
        std::vector<int> indicies_found;
        std::vector<float> squaredDistances;
        kdTree.radiusSearch(i, radius, indicies_found, squaredDistances, 2);

        int num_edges = indicies_found.size() - 1;
        // If the minimum edges end up equaling 1 then it is a disconnected graph.
        // reset the laplacian, reset the counter and increase the radius until we
        // have a fully connected graph
        if (num_edges == 0)
        {
            std::cout << "Increasing radius" << std::endl;
            i = 0;
            radius += 0.01f;
        }
    }

    std::cout << "Radius: " << radius << std::endl;
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
    unsigned int max_nn = 100;

    std::cout << "Laplacia3 size: " << size << std::endl;
    for (int i = 0; i < size; i++) {
        std::vector<int> indicies_found;
        std::vector<float> squaredDistances;
        kdTree.radiusSearch(i, radius, indicies_found, squaredDistances, max_nn);

        int num_edges = indicies_found.size() - 1;
        laplacian(i,i) = num_edges;

        for (int j = 1; j < indicies_found.size(); j++)
        {
            laplacian(i, indicies_found[j]) = -1; // / (sqrt(squaredDistances[j]) + bias);
            laplacian(indicies_found[j], i) = -1; // / (sqrt(squaredDistances[j]) + bias);
        }
    }
}

void ComputeEigensSparse(arma::sp_mat &laplacian, arma::vec &eigval, arma::mat &eigvec)
{
    auto start = std::chrono::steady_clock::now();
    arma::eigs_sym(eigval, eigvec, laplacian, 100);
    auto end = std::chrono::steady_clock::now();
    cout << "Elapsed time in miliseconds: "
        << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
        << " ms" << endl;
}

double ComputeMinRadius(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, pcl::KdTreeFLANN<pcl::PointXYZ> &kdTree)
{
    kdTree.setInputCloud(cloud);

    int rows = cloud->points.size();

    double radius = FindMinRadius(kdTree, rows);

    return radius;
}

void ComputeLaplacian(arma::sp_mat &laplacian, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::KdTreeFLANN<pcl::PointXYZ> &kdTree, double radius)
{
    int rows = cloud->points.size();
    int cols = rows;

    laplacian = arma::sp_mat(rows, cols);

    PopulateLaplacian(kdTree, laplacian, rows, radius);
}

void VisualizeHistograms(arma::vec &ref_eigval, arma::vec &query_eigval)
{
    auto f = figure(true);
    f->width(f->width() * 3);
    f->height(f->height() * 2.5);
    f->x_position(10);
    f->y_position(10);

    std::vector<double> ref = arma::conv_to< std::vector<double> >::from(ref_eigval);
    std::vector<double> query = arma::conv_to< std::vector<double> >::from(query_eigval);

    int r_size = ref.size();
    //double min_ref = *std::min_element(eigval_vec.begin(), eigval_vec.end());
    //double max_eigval = *std::max_element(eigval_vec.begin(), eigval_vec.end());

    //double bin_width = (max_eigval - min_eigval) / 25;

    auto h1 = hist(ref);
    h1->face_color("r");
    h1->edge_color("r");
    //h1->bin_width(.3);
    hold(on);
    auto h2 = hist(query);
    h2->face_color("b");
    h2->edge_color("b");
    //h2->bin_width(.3);
    title("Cubes");
    f->draw();
    show();
}

void VisualizeHistogram(arma::vec &eigval, arma::mat &eigvec)
{
    auto f = figure(true);
    f->width(f->width() * 3);
    f->height(f->height() * 2.5);
    f->x_position(10);
    f->y_position(10);

    std::vector<double> eigval_vec = arma::conv_to< std::vector<double> >::from(eigval);


    int r_size = eigval_vec.size();
    //double min_eigval = *std::min_element(eigval_vec.begin(), eigval_vec.end());
    //double max_eigval = *std::max_element(eigval_vec.begin(), eigval_vec.end());

    //double bin_width = (max_eigval - min_eigval) / 25;

    auto h1 = hist(eigval_vec);
    //h1->bin_width(bin_width);
    title("TeddyBear");
    f->draw();
    show();
}

void CreateCubePointCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr &cube, pcl::PointCloud<pcl::PointXYZ>::Ptr &dense_cube)
{
    cube->reserve(25 * 6);
    dense_cube->reserve(100 * 6);

    // Regular cube
    for(int x = 0; x < 5; x++)
    {
        for(int y = 0; y < 5; y++)
        {
            //Bottom
            pcl::PointXYZ bottom(x, y, 0);
            cube->push_back(bottom);
            //Top
            pcl::PointXYZ top(x, y, 4);
            cube->push_back(top);
            //Left
            pcl::PointXYZ left(0, x, y);
            cube->push_back(left);
            //Back
            pcl::PointXYZ back(x, 4, y);
            cube->push_back(back);
            //Right
            pcl::PointXYZ right(4, x, y);
            cube->push_back(right);
            //Front
            pcl::PointXYZ front(x, 0, y);
            cube->push_back(front);
        }
    }

    // dense cube
    for(int x = 0; x < 9; x++)
    {
        for(int y = 0; y < 9; y++)
        {
            //Bottom
            pcl::PointXYZ bottom(.5*x, .5*y, 0);
            dense_cube->push_back(bottom);
            //Top
            pcl::PointXYZ top(.5*x, .5*y, 4);
            dense_cube->push_back(top);
            //Left
            pcl::PointXYZ left(0, .5*x, .5*y);
            dense_cube->push_back(left);
            //Back
            pcl::PointXYZ back(.5*x, 4, .5*y);
            dense_cube->push_back(back);
            //Right
            pcl::PointXYZ right(4, .5*x, .5*y);
            dense_cube->push_back(right);
            //Front
            pcl::PointXYZ front(.5*x, 0, .5*y);
            dense_cube->push_back(front);
        }
    }
}



int main()
{
    // TODO
    std::string ref_scan= "/home/nate/Downloads/tablesPly/b1.ply";
    //std::string ref_scan= "/home/nate/Development/3RScan/data/3RScan/4acaebcc-6c10-2a2a-858b-29c7e4fb410d/labels.instances.annotated.v2.ply";
    //std::string query_scan= "/home/nate/Development/3RScan/data/3RScan/754e884c-ea24-2175-8b34-cead19d4198d/labels.instances.annotated.v2.ply";
    std::string query_scan= "/home/nate/Datasets/tablesPly/b1.ply";
    pcl::PointCloud<pcl::PointXYZ>::Ptr ref_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr query_cloud(new pcl::PointCloud<pcl::PointXYZ>);

    pcl::KdTreeFLANN<pcl::PointXYZ> ref_kdTree;
    pcl::KdTreeFLANN<pcl::PointXYZ> query_kdTree;


    arma::sp_mat ref_laplacian, query_laplacian;
    arma::vec ref_eigval, query_eigval;
    arma::mat ref_eigvec, query_eigvec;

    //ReadPointCloud(teddyBearFile, cloud);
    //CreateCubePointCloud(cube, dense_cube);
    ReadPointCloud(ref_scan, ref_cloud);
    ReadPointCloud(query_scan, query_cloud);
    VisualizeCloud(ref_cloud);
    VisualizeCloud(query_cloud);

    double ref_radius = ComputeMinRadius(ref_cloud, ref_kdTree);
    double query_radius = ComputeMinRadius(query_cloud, query_kdTree);
    double radius = std::max(ref_radius, query_radius);
    std::cout << "Radius used: " << radius << std::endl;

    ComputeLaplacian(ref_laplacian, ref_cloud, ref_kdTree, radius);
    ComputeLaplacian(query_laplacian, query_cloud, query_kdTree, radius);

    //ComputeLaplacian(dense_laplacian, dense_cube);
    ComputeEigensSparse(ref_laplacian, ref_eigval, ref_eigvec);
    ComputeEigensSparse(query_laplacian, query_eigval, query_eigvec);


    std::cout << "Cube eig" << std::endl;
    std::cout << ref_eigval << std::endl;
    std::cout << "Dense eig" << std::endl;
    std::cout << query_eigval << std::endl;
    VisualizeHistograms(ref_eigval, query_eigval);
    //VisualizeHistogram(cube_eigval, cube_eigvec);
    //VisualizeHistogram(dense_eigval, dense_eigvec);
    //std::cout << "eigval: " << eigval << std::endl;
    //std::cout << "eigvec rows: " << eigvec.col(0) << std::endl;
    //std::cout << laplacian.row(0) << std::endl;
    //std::cout << laplacian.row(0) * eigvec.col(0) << std::endl;
    //// plt eigenvectors in histogram
    return 1;
}
