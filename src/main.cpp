#include <iostream>
#include <thread>
#include <chrono>
#include <sstream>
#include <fstream>

#include <pcl/io/ply_io.h>
#include <pcl/conversions.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/filters/conditional_removal.h>
#include <pcl/filters/random_sample.h>

#include <boost/format.hpp>

#include <nlohmann/json.hpp>

#include <armadillo>

using namespace std::chrono_literals;
using json = nlohmann::json;

struct SpectralFeatures
{
    std::string ply_file;
    std::string scan_id;
    std::unordered_map<std::string, json> obj_details_map;
    std::unordered_map<std::string,arma::sp_mat> laplacian_map;
    std::unordered_map<std::string,arma::vec> eigval_map;
    std::unordered_map<std::string,arma::mat> eigvec_map;
    std::unordered_map<std::string, pcl::PointCloud<pcl::PointXYZRGB>::Ptr> obj_cloud_map;
};

void ComputeLaplacian(SpectralFeatures &sf)
{
    for( auto const kv : sf.obj_cloud_map )
    {
        pcl::KdTreeFLANN<pcl::PointXYZRGB> kdTree;
        kdTree.setInputCloud(kv.second);
        pcl::PointXYZ searchPoint;

        std::vector<int> indicies_found;
        std::vector<float> squaredDistances;

        int rows = kv.second->points.size();
        int cols = rows;

        int K = 5;

        arma::sp_mat laplacian(rows, cols);

        std::cout << "Laplacian size: " << rows << std::endl;
        for (int i = 0; i < rows; i++) {
            kdTree.nearestKSearch(i, K, indicies_found, squaredDistances);

            int num_edges = indicies_found.size();
            laplacian(i,i) = num_edges - 1;

            for (int j = 1; j < indicies_found.size(); j++)
            {
                laplacian(i, indicies_found[j]) = -1 * sqrt(squaredDistances[j]);
                laplacian(indicies_found[j], i) = -1 * sqrt(squaredDistances[j]);
            }
        }

        sf.laplacian_map[kv.first] = laplacian;
    }
}

void ComputeEigens(SpectralFeatures &sf)
{
    for( auto const kv : sf.laplacian_map)
    {
        arma::vec eigval;
        arma::mat eigvec;

        auto start = std::chrono::steady_clock::now();
        arma::eigs_sym(eigval, eigvec, kv.second, kv.second.n_rows - 1);
        arma::vec sorted_eig = arma::sort(eigval);
        auto end = std::chrono::steady_clock::now();
        cout << "Elapsed time in miliseconds: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
            << " ms" << endl;

        sf.eigval_map[kv.first] = sorted_eig;
        sf.eigvec_map[kv.first] = eigvec;
    }
}

void ParseConfig(SpectralFeatures &sf)
{
    std::string objConfig = "/home/nate/Development/3RScan/data/3RScan/objects.json";
    std::ifstream f(objConfig);
    json data = json::parse(f);

    std::vector<json> objects;
    for (auto obj : data["scans"])
    {
        if( obj["scan"] == sf.scan_id )
        {
            objects = obj["objects"];
            break;
        }
    }

    // Fill map with the colors
    for (auto const obj: objects)
    {
        std::string ply_color = obj["ply_color"];
        ply_color.erase(0, 1);
        sf.obj_details_map[ply_color] = obj;
    }
}

void VisualizeCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud)
{
    pcl::visualization::CloudViewer viewer("Cloud Viewer");
    viewer.showCloud(cloud);
    std::cout << '\n' << "Press Enter";
    while (std::cin.get() != '\n') {}
}


void PopulateSemanticEigenMap(SpectralFeatures &sf)
{
    pcl::PCLPointCloud2::Ptr cloud2(new pcl::PCLPointCloud2 ());
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZRGB>);

    pcl::io::loadPLYFile(sf.ply_file, *cloud2);
    pcl::fromPCLPointCloud2(*cloud2, *cloud);

    std::cout << sf.obj_details_map.size() << std::endl;
    for ( auto const &kv : sf.obj_details_map)
    {
        std::cout << kv.second["label"] << std::endl;
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
        condrem.filter(*cloud_filtered);

        // Compute Random Sample so that we have 300 pts for each object
        std::cout << "Original Obj size: " << cloud_filtered->points.size() << std::endl;
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr rand_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
        pcl::RandomSample <pcl::PointXYZRGB> random;

        random.setInputCloud(cloud_filtered);
        random.setSeed (std::rand ());
        random.setSample((unsigned int)(300));
        random.filter(*rand_cloud);

        sf.obj_cloud_map[kv.first] = rand_cloud;

        //VisualizeCloud(rand_cloud);
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

void DownSizeMatchClouds(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud, int size)
{
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr rand_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::RandomSample <pcl::PointXYZRGB> random;

    random.setInputCloud(cloud);
    random.setSeed (std::rand ());
    random.setSample(size);
    random.filter(*rand_cloud);

    cloud = rand_cloud;
}

void NormalizeObjClouds(std::unordered_map<std::string, pcl::PointCloud<pcl::PointXYZRGB>::Ptr> &larger, std::unordered_map<std::string, pcl::PointCloud<pcl::PointXYZRGB>::Ptr> &smaller)
{
    for ( auto kv : larger )
    {
        if (smaller.find(kv.first) != smaller.end())
        {
            int size = std::min(larger[kv.first]->points.size(), smaller[kv.first]->points.size());
            if (larger[kv.first]->points.size() > smaller[kv.first]->points.size())
                DownSizeMatchClouds(larger[kv.first], size);
            else if (larger[kv.first]->points.size() > smaller[kv.first]->points.size())
                DownSizeMatchClouds(smaller[kv.first], size);
            std::cout << "temp" << std::endl;
        }
    }
}


int main()
{
    SpectralFeatures ref_sf;
    SpectralFeatures query_sf;

    std::string ref_scan = "/home/nate/Development/3RScan/data/3RScan/4acaebcc-6c10-2a2a-858b-29c7e4fb410d/labels.instances.annotated.v2.ply";
    std::string query_scan = "/home/nate/Development/3RScan/data/3RScan/754e884c-ea24-2175-8b34-cead19d4198d/labels.instances.annotated.v2.ply";

    std::string ref_scan_id = "4acaebcc-6c10-2a2a-858b-29c7e4fb410d";
    std::string query_scan_id = "754e884c-ea24-2175-8b34-cead19d4198d";

    ref_sf.ply_file = ref_scan;
    ref_sf.scan_id = ref_scan_id;

    query_sf.ply_file = query_scan;
    query_sf.scan_id = query_scan_id;

    ParseConfig(ref_sf);
    ParseConfig(query_sf);

    PopulateSemanticEigenMap(ref_sf);
    PopulateSemanticEigenMap(query_sf);

    // Create new function which chooses random on larger cloud in order to get the clouds the same size

    if ( ref_sf.obj_details_map.size() > query_sf.obj_details_map.size() )
        NormalizeObjClouds(ref_sf.obj_cloud_map, query_sf.obj_cloud_map);
    else
        NormalizeObjClouds(query_sf.obj_cloud_map, ref_sf.obj_cloud_map);

    ComputeLaplacian(ref_sf);
    ComputeLaplacian(query_sf);

    ComputeEigens(ref_sf);
    ComputeEigens(query_sf);

    //std::cout << "ref map size: " << ref_semantic_eigen_map.size() << std::endl;
    //std::cout << "query map size: " << query_semantic_eigen_map.size() << std::endl;

    for ( auto const kv : ref_sf.eigval_map )
    {
        if (query_sf.eigval_map.find(kv.first) != query_sf.eigval_map.end())
        {
            std::cout << "color: " << kv.first << " Number of eigenvalues: " << kv.second.size() << std::endl;
            std::cout << "color: " << kv.first << " Number of eigenvalues: " << query_sf.eigval_map[kv.first].size() << std::endl;
            std::cout << "!!!!!!!!!!!!!!!" << std::endl;
        }
        else
        {
            std::cout << "ABSENT OBJECT" << std::endl;
            std::cout << "Ref color: " << kv.first << " Number of eigenvalues: " << kv.second.size() << std::endl;

        }
    }

    std::cout << "SUCCESS" << std::endl;

    return 1;
}
