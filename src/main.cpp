#include <iostream>
#include <thread>
#include <chrono>
#include <sstream>

#include <pcl/io/ply_io.h>
#include <pcl/conversions.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/filters/conditional_removal.h>

#include <boost/format.hpp>

#include <armadillo>

using namespace std::chrono_literals;

std::unordered_map<std::string, std::vector<double>> semantic_eigen_map;

void VisualizeCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud)
{
    pcl::visualization::CloudViewer viewer("Cloud Viewer");
    viewer.showCloud(cloud);
    do
    {
        std::cout << '\n' << "Press a key to continue...";
    } while (std::cin.get() != '\n');
}


void PopulateSemanticEigenMap(std::string plyFile)
{
    pcl::PCLPointCloud2::Ptr cloud2(new pcl::PCLPointCloud2 ());
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZRGB>);

    pcl::io::loadPLYFile(plyFile, *cloud2);
    pcl::fromPCLPointCloud2(*cloud2, *cloud);

    for ( auto const &data : cloud->points )
    {
        std::stringstream ss;
        ss << boost::format("%x%x%x") % unsigned(data.r) % unsigned(data.g) % unsigned(data.b);
        std::string key = ss.str();
        semantic_eigen_map[key] = std::vector<double>();
    }

    for ( auto const &kv : semantic_eigen_map )
    {
        uint8_t r = (uint8_t) strtol(kv.first.substr(0,2).c_str(), nullptr, 16);
        uint8_t g = (uint8_t) strtol(kv.first.substr(2,2).c_str(), nullptr, 16);
        uint8_t b = (uint8_t) strtol(kv.first.substr(4,2).c_str(), nullptr, 16);

        pcl::ConditionAnd<pcl::PointXYZRGB>::Ptr range_cond(new pcl::ConditionAnd<pcl::PointXYZRGB>());
        range_cond->addComparison(pcl::PackedRGBComparison<pcl::PointXYZRGB>::ConstPtr(new pcl::PackedRGBComparison<pcl::PointXYZRGB>("r", pcl::ComparisonOps::EQ, r)));
        range_cond->addComparison(pcl::PackedRGBComparison<pcl::PointXYZRGB>::ConstPtr(new pcl::PackedRGBComparison<pcl::PointXYZRGB>("g", pcl::ComparisonOps::EQ, g)));
        range_cond->addComparison(pcl::PackedRGBComparison<pcl::PointXYZRGB>::ConstPtr(new pcl::PackedRGBComparison<pcl::PointXYZRGB>("b", pcl::ComparisonOps::EQ, b)));

        pcl::ConditionalRemoval<pcl::PointXYZRGB> condrem;
        condrem.setCondition(range_cond);
        condrem.setInputCloud(cloud);
        condrem.setKeepOrganized(true);
        condrem.filter(*cloud_filtered);

        //VisualizeCloud(cloud_filtered);
    }


}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr ParseAndFilter(std::string plyFile)
{
    pcl::PCLPointCloud2::Ptr cloud2(new pcl::PCLPointCloud2 ());
    pcl::PCLPointCloud2::Ptr cloud2_filtered(new pcl::PCLPointCloud2 ());
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr ret_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::io::loadPLYFile(plyFile, *cloud2);

    pcl::VoxelGrid<pcl::PCLPointCloud2> voxGrid;
    voxGrid.setInputCloud(cloud2);
    voxGrid.setLeafSize (1.0f, 1.0f, 1.0f);
    voxGrid.filter(*cloud2_filtered);

    pcl::fromPCLPointCloud2(*cloud2_filtered, *ret_cloud);
    return ret_cloud;
}

void ComputeLaplacian(arma::sp_mat &laplacian, pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud)
{
    pcl::KdTreeFLANN<pcl::PointXYZRGB> kdTree;
    kdTree.setInputCloud(cloud);
    pcl::PointXYZ searchPoint;

    std::vector<int> indicies_found;
    std::vector<float> radiusSquaredDistance;
    int max_nn = 20;

    float radius = 1.2f;

    int rows = cloud->size();
    int cols = rows;

    laplacian.set_size(rows, cols);

    std::cout << "Laplacian size: " << rows << std::endl;
    for (int i = 0; i < rows; i++) {
        kdTree.radiusSearch(i, radius, indicies_found, radiusSquaredDistance, max_nn);

        // Populate the laplacian diaganol w/ edge count
        int num_edges = indicies_found.size();
        //std::cout << "Edges found: " << num_edges << std::endl;
        laplacian(i,i) = num_edges - 1;

        // Populate the adjacency Matrix with -1 where edge is found
        for (int j = 1; j < indicies_found.size(); j++)
        {
            laplacian(i, indicies_found[j]) = -1 * sqrt(radiusSquaredDistance[j]);
            // Since laplcian is symmetric populare the j, i as well
            laplacian(indicies_found[j], i) = -1 * sqrt(radiusSquaredDistance[j]);
        }
    }
}

void ComputeEigens(arma::vec &eigval, arma::mat &eigvec, arma::sp_mat &laplacian, int size)
{
    auto start = std::chrono::steady_clock::now();
    arma::eigs_sym(eigval, eigvec, laplacian, size - 1);
    arma::vec sorted_eig = arma::sort(eigval);
    auto end = std::chrono::steady_clock::now();
    cout << "Elapsed time in miliseconds: "
        << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
        << " ms" << endl;
}

int main()
{
    std::string ref_scan = "/home/nate/Development/3RScan/data/3RScan/4acaebcc-6c10-2a2a-858b-29c7e4fb410d/labels.instances.annotated.v2.ply";
    std::string query_scan = "/home/nate/Development/3RScan/data/3RScan/754e884c-ea24-2175-8b34-cead19d4198d/labels.instances.annotated.v2.ply";

    //pcl::PointCloud<pcl::PointXYZRGB>::Ptr ref_cloud = ParseAndFilter(ref_scan);
    //pcl::PointCloud<pcl::PointXYZRGB>::Ptr query_cloud = ParseAndFilter(query_scan);

    // TODO
    //  1) Filter point cloud to get each object
    //      a) create map with color as key and array of eigenvalues for values


    PopulateSemanticEigenMap(ref_scan);

    //VisualizeCloud(ref_cloud);
    //VisualizeCloud(query_cloud);

    //arma::sp_mat ref_laplacian;
    //arma::sp_mat query_laplacian;

    //ComputeLaplacian(ref_laplacian, ref_cloud);
    //ComputeLaplacian(query_laplacian, query_cloud);

    //arma::vec ref_eigval, query_eigval;
    //arma::mat ref_eigvec, query_eigvec;

    //ComputeEigens(ref_eigval, ref_eigvec, ref_laplacian, ref_cloud->size());
    //ComputeEigens(query_eigval, query_eigvec, query_laplacian, query_cloud->size());

    //int len = std::max(ref_eigval.size(), query_eigval.size());

    //int diff = 0;
    //for(int i = 0; i < len; i++)
    //{
    //    std::cout << "Ref: " << ref_eigval[i] << " Query: " << query_eigval[i] << std::endl;
    //    diff += abs( ref_eigval[i] - query_eigval[i] );
    //}


    return 1;
}
