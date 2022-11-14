#include <iostream>
#include <thread>
#include <chrono>
#include <sstream>
#include <fstream>
#include <random>
#include <cmath>

#include <pcl/common/centroid.h>
#include <pcl/io/ply_io.h>
#include <pcl/conversions.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/filters/conditional_removal.h>
#include <pcl/filters/random_sample.h>
#include <pcl/common/common.h>

#include <boost/format.hpp>

#include <nlohmann/json.hpp>

#include <matplot/matplot.h>

#include <armadillo>

#include <cfenv>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

using namespace std::chrono_literals;
using namespace matplot;
using json = nlohmann::json;

struct SpectralFeaturesDense
{
    std::string ply_file;
    std::string scan_id;
    std::unordered_map<std::string, json> obj_details_map;
    std::unordered_map<std::string,arma::mat> laplacian_map;
    std::unordered_map<std::string,arma::vec> eigval_map;
    std::unordered_map<std::string,arma::mat> eigvec_map;
    std::unordered_map<std::string, pcl::PointCloud<pcl::PointXYZRGB>::Ptr> obj_cloud_map;
};

struct SpectralFeaturesSparse
{
    std::string ply_file;
    std::string scan_id;
    std::string referece_id; // Only populared by SF's in the query scan map
    std::unordered_map<std::string, json> obj_details_map;
    std::unordered_map<std::string,arma::sp_mat> laplacian_map;
    std::unordered_map<std::string,arma::vec> eigval_map;
    std::unordered_map<std::string,arma::mat> eigvec_map;
    std::unordered_map<std::string, pcl::PointCloud<pcl::PointXYZRGB>::Ptr> obj_cloud_map;
};

// NOTE: for gdb debugging
std::string make_string(const char *x)
{
        return x;
}

template<class Matrix>
void print_matrix(Matrix matrix) {
    matrix.print(std::cout);
}

template void print_matrix<arma::mat>(arma::mat matrix);

double GetSmallestDistance(pcl::KdTreeFLANN<pcl::PointXYZRGB> &kdTree, int size)
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

template<class T>
void ComputeLaplacianFC(T &sf)
{
    for( auto const kv : sf.obj_cloud_map )
    {
        pcl::KdTreeFLANN<pcl::PointXYZRGB> kdTree;
        kdTree.setInputCloud(kv.second);
        pcl::PointXYZ searchPoint;


        int rows = kv.second->points.size();
        int cols = rows;

        double radius = 1000;
        unsigned int max_nn = rows + 1;

        double smallestDistance = GetSmallestDistance(kdTree, rows);
        double bias = 1.0 - smallestDistance;

        arma::mat laplacian(rows, cols);

        std::cout << "Laplacian size: " << rows << std::endl;
        for (int i = 0; i < rows; i++) {
            std::vector<int> indicies_found;
            std::vector<float> squaredDistances;
            kdTree.radiusSearch(i, radius, indicies_found, squaredDistances, max_nn);

            int num_edges = indicies_found.size() - 1;
            if (num_edges != rows - 1)
                std::cout << "Did not get all points!" << std::endl;
            //std::cout << num_edges << std::endl;
            laplacian(i,i) = num_edges;

            for (int j = 1; j < indicies_found.size(); j++)
            {
                laplacian(i, indicies_found[j]) = (-1 / (sqrt(squaredDistances[j]) + bias));
                laplacian(indicies_found[j], i) = -1 / (sqrt(squaredDistances[j]) + bias);
                //laplacian(i, indicies_found[j]) = -1;
                //laplacian(indicies_found[j], i) = -1;
            }
        }

        //std::cout << laplacian << std::endl;
        arma::rowvec A = max(laplacian,0);
        for ( auto & val : A ) if ( val == A.size() - 1 ) val = 0;
        std::cout << "max row: " << max(A) << std::endl;
        std::cout << "min row: " << min(A) << std::endl;
        sf.laplacian_map[kv.first] = laplacian;
    }
}

double FindMinRadius(pcl::KdTreeFLANN<pcl::PointXYZRGB> &kdTree, int size)
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

void PopulateLaplacian(pcl::KdTreeFLANN<pcl::PointXYZRGB> &kdTree, arma::sp_mat &laplacian, int size, double radius)
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

void ComputeLaplacian(SpectralFeaturesSparse &ref_sf, SpectralFeaturesSparse &query_sf)
{
    for( auto const kv : ref_sf.obj_details_map)
    {
        if (query_sf.obj_cloud_map.find(kv.first) != query_sf.obj_cloud_map.end())
        {
            pcl::KdTreeFLANN<pcl::PointXYZRGB> ref_kdTree;
            pcl::KdTreeFLANN<pcl::PointXYZRGB> query_kdTree;
            ref_kdTree.setInputCloud(ref_sf.obj_cloud_map[kv.first]);
            query_kdTree.setInputCloud(query_sf.obj_cloud_map[kv.first]);

            int ref_rows = ref_sf.obj_cloud_map[kv.first]->points.size();
            int query_rows = query_sf.obj_cloud_map[kv.first]->points.size();


            double radius = std::max(FindMinRadius(ref_kdTree, ref_rows), FindMinRadius(query_kdTree, query_rows));

            arma::sp_mat ref_laplacian(ref_rows, ref_rows);
            arma::sp_mat query_laplacian(query_rows, query_rows);

            PopulateLaplacian(ref_kdTree, ref_laplacian, ref_rows, radius);
            PopulateLaplacian(query_kdTree, query_laplacian, query_rows, radius);

            ref_sf.laplacian_map[kv.first] = ref_laplacian;
            query_sf.laplacian_map[kv.first] = query_laplacian;
        }
    }
}

void ComputeLaplacianKnn(SpectralFeaturesSparse &sf)
{
    for( auto const kv : sf.obj_cloud_map )
    {
        pcl::KdTreeFLANN<pcl::PointXYZRGB> kdTree;
        kdTree.setInputCloud(kv.second);
        pcl::PointXYZ searchPoint;


        int rows = kv.second->points.size();
        int cols = rows;

        double neighbors = 6;

        double smallestDistance = GetSmallestDistance(kdTree, rows);
        double bias = 1.0 - smallestDistance;

        arma::sp_mat laplacian(rows, cols);

        std::cout << "Laplacian size: " << rows << std::endl;
        for (int i = 0; i < rows; i++) {
            std::vector<int> indicies_found;
            std::vector<float> squaredDistances;
            kdTree.nearestKSearch(kv.second->points[i], neighbors, indicies_found, squaredDistances);

            int num_edges = indicies_found.size() - 1;
            if (num_edges != neighbors - 1)
                std::cout << "Did not get the correct number of neighbors" << std::endl;
            //std::cout << num_edges << std::endl;
            laplacian(i,i) = num_edges;

            for (int j = 1; j < indicies_found.size(); j++)
            {
                laplacian(i, indicies_found[j]) = (-1 / (sqrt(squaredDistances[j]) + bias));
                laplacian(indicies_found[j], i) = -1 / (sqrt(squaredDistances[j]) + bias);
                //laplacian(i, indicies_found[j]) = -1;
                //laplacian(indicies_found[j], i) = -1;
            }
        }

        //std::cout << laplacian << std::endl;
       // arma::rowvec A = max(laplacian,0);
       // for ( auto & val : A ) if ( val == A.size() - 1 ) val = 0;
       // std::cout << "max row: " << max(A) << std::endl;
       // std::cout << "min row: " << min(A) << std::endl;
        sf.laplacian_map[kv.first] = laplacian;
    }
}

void ComputeEigensDense(SpectralFeaturesDense &sf)
{
    for( auto const kv : sf.laplacian_map)
    {
        arma::vec eigval;
        arma::mat eigvec;

        auto start = std::chrono::steady_clock::now();
        arma::eig_sym(eigval, eigvec, kv.second);
        //arma::vec sorted_eig = arma::sort(eigval);
        auto end = std::chrono::steady_clock::now();
        cout << "Elapsed time in miliseconds: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
            << " ms" << endl;

        sf.eigval_map[kv.first] = eigval;
        sf.eigvec_map[kv.first] = eigvec;
    }
}

void ComputeEigensSparse(SpectralFeaturesSparse &sf)
{
    for( auto const kv : sf.laplacian_map)
    {
        arma::vec eigval;
        arma::mat eigvec;

        auto start = std::chrono::steady_clock::now();
        arma::eigs_sym(eigval, eigvec, kv.second, kv.second.n_cols - 1);
        auto end = std::chrono::steady_clock::now();
        cout << "Elapsed time in miliseconds: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
            << " ms" << endl;

        sf.eigval_map[kv.first] = eigval;
        sf.eigvec_map[kv.first] = eigvec;
    }
}

template<class T>
void ComputeEigensSparse(T &sf)
{
    for( auto const kv : sf.laplacian_map)
    {
        arma::vec eigval;
        arma::mat eigvec;

        auto start = std::chrono::steady_clock::now();
        arma::eigs_sym(eigval, eigvec, kv.second);
        //arma::vec sorted_eig = arma::sort(eigval);
        auto end = std::chrono::steady_clock::now();
        cout << "Elapsed time in miliseconds: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
            << " ms" << endl;

        sf.eigval_map[kv.first] = eigval;
        sf.eigvec_map[kv.first] = eigvec;
    }
}

//template<class T>
//void ParseConfig(T &sf)
void ParseConfig(std::unordered_map<std::string, SpectralFeaturesSparse> &reference_map, std::unordered_map<std::string, SpectralFeaturesSparse> &query_map)
{
    std::string config = "/home/nate/Development/SGPR/config/config.json";
    std::string objConfig = "/home/nate/Development/SGPR/config/objects.json";
    std::string refQueryMapConfig = "/home/nate/Development/SGPR/config/3RScan.json";

    std::ifstream configFile(config);
    std::ifstream objectFile(objConfig);
    std::ifstream refQueryMapFile(refQueryMapConfig);

    json configData = json::parse(configFile);
    json objectData = json::parse(objectFile);
    json refQueryMapData = json::parse(refQueryMapFile);

    std::string dataset_dir = configData["dataset_dir"].get<std::string>();
    std::string ply_file = configData["ply_filename"].get<std::string>();
    for (auto const &reference_scan: configData["reference_scans"])
    {
        SpectralFeaturesSparse sf;
        sf.scan_id = reference_scan["scan_id"].get<std::string>();
        sf.ply_file = dataset_dir + "/" + sf.scan_id + "/" + ply_file;

        // Get all the objects associated with the scan
        std::vector<json> objects;
        for (auto obj : objectData["scans"])
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

        reference_map[sf.scan_id] = sf;

        // For each reference scan we need to find all the query scans and populare the queryscan map
        for (auto const &jsonBlock : refQueryMapData["scans"])
        {
            if (jsonBlock["reference"] == sf.scan_id)
            {
                std::vector<json> query_scans = jsonBlock["scans"];
                for (auto const &queryScanBlock : query_scans)
                {
                    SpectralFeaturesSparse query_sf;
                    query_sf.scan_id = queryScanBlock["reference"].get<std::string>();
                    query_sf.ply_file = dataset_dir + "/" + query_sf.scan_id + "/" + ply_file;

                    // Get all the objects associated with the scan
                    std::vector<json> objects;
                    for (auto obj : objectData["scans"])
                    {
                        if( obj["scan"] == query_sf.scan_id )
                        {
                            objects = obj["objects"];
                            break;
                        }
                    }

                    for (auto const obj: objects)
                    {
                        std::string ply_color = obj["ply_color"];
                        ply_color.erase(0, 1);
                        query_sf.obj_details_map[ply_color] = obj;
                    }

                    query_sf.referece_id = sf.scan_id;
                    query_map[query_sf.scan_id] = query_sf;
                }
            }
        }
    }
}

void VisualizeCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud)
{
    pcl::visualization::CloudViewer viewer("Cloud Viewer");
    viewer.showCloud(cloud);
    std::cout << '\n' << "Press Enter";
    while (std::cin.get() != '\n') {}
}


template<class T>
void PopulateSemanticEigenMap(T &sf)
{
    pcl::PCLPointCloud2::Ptr cloud2(new pcl::PCLPointCloud2 ());
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered;

    pcl::io::loadPLYFile(sf.ply_file, *cloud2);
    pcl::fromPCLPointCloud2(*cloud2, *cloud);

    std::cout << sf.obj_details_map.size() << std::endl;
    for ( auto &kv : sf.obj_details_map)
    {
        cloud_filtered = boost::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();
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
        double leafInc = 0.01;
        double leafValue = 0.01;
        //pcl::PointCloud<pcl::PointXYZRGB>::Ptr input_cloud = cloud_filtered;
        while (cloud_filtered->points.size() > 1000)
        {
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr vox_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
            pcl::VoxelGrid<pcl::PointXYZRGB> sor;
            sor.setInputCloud (cloud_filtered);
            sor.setLeafSize (leafValue, leafValue, leafValue);
            sor.filter (*vox_cloud);

            cloud_filtered = vox_cloud;
            leafValue += leafInc;
        }

        std::cout << "The cloud which will be saved is of size: " << cloud_filtered->points.size() << std::endl;

        //pcl::RandomSample <pcl::PointXYZRGB> random;

        //random.setInputCloud(cloud_filtered);
        //random.setSeed (std::rand ());
        //random.setSample((unsigned int)(1000));
        //random.filter(*rand_cloud);

        sf.obj_cloud_map[kv.first] = cloud_filtered;
        cloud_filtered.reset();

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

    cloud.reset();
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
            else if (larger[kv.first]->points.size() < smaller[kv.first]->points.size())
                DownSizeMatchClouds(smaller[kv.first], size);
        }
    }
}

// TODO I would like to find a way to get the same number of eigenvalues
// Figure out a removal algorithm to remove points based on binning
template<class T>
void VisualizeHistograms(T &ref_sf, T &query_sf)
{

    auto f = figure(true);
    f->width(f->width() * 3);
    f->height(f->height() * 2.5);
    f->x_position(10);
    f->y_position(10);
    int row_idx = 0, col_idx = 0;
    for ( auto kv : ref_sf.obj_details_map)
    {
        if (query_sf.eigval_map.find(kv.first) != query_sf.eigval_map.end())
        {
            std::cout << kv.second["label"] << std::endl;
            std::cout << row_idx << " : " << col_idx << std::endl;
            std::vector<double> ref = arma::conv_to< std::vector<double> >::from(ref_sf.eigval_map[kv.first]);
            std::vector<double> query = arma::conv_to< std::vector<double> >::from(query_sf.eigval_map[kv.first]);


            int r_size = ref.size();
            int q_size = query.size();
            std::cout << "ref size: " << r_size << std::endl;
            std::cout << "query size: " << q_size << std::endl;
            if(ref.size() != query.size())
                std::cout << "DOES NOT EQUAL" << std::endl;

            //std::vector<double> ref_filtered;
            //std::vector<double> query_filtered;

            //auto filter_outliers = [](double i) {
            //    return (i > 0.5);
            //};

            //std::copy_if(ref.begin(), ref.end(), std::back_inserter(ref_filtered), filter_outliers);
            //std::copy_if(query.begin(), query.end(), std::back_inserter(query_filtered), filter_outliers);

            // Get min and max value in both vectors
            double min_ref = *std::min_element(ref.begin(), ref.end());
            double max_ref = *std::max_element(ref.begin(), ref.end());
            double min_query = *std::min_element(query.begin(), query.end());
            double max_query = *std::max_element(query.begin(), query.end());

            double min = std::min(min_ref, min_query);
            double max = std::max(max_ref, max_query);

            double bin_width = (max - min) / 25;

            std::string label = kv.second["label"].dump() + " : " + kv.first;
            subplot(6,6,row_idx * 6 + col_idx);
            auto h1 = hist(ref);
            h1->face_color("r");
            h1->edge_color("r");
            hold(on);
            auto h2 = hist(query);
            h2->face_color("b");
            h2->edge_color("b");
            h1->bin_width(bin_width);
            h2->bin_width(bin_width);
            title(label);
            f->draw();

            // Visualize the cloud
            //VisualizeCloud(ref_sf.obj_cloud_map[kv.first]);
            //VisualizeCloud(query_sf.obj_cloud_map[kv.first]);

            //show();
            //cla();

            //subplot(1,2,1);
            //hist(query);
            //std::string title2 = "Query " + kv.second["label"].dump();
            //title(title2);

            if (col_idx == 5)
            {
                row_idx++;
                col_idx = 0;
            }
            else
                col_idx++;

        }
    }
    show();
    cla();
}

void FullyConnected(SpectralFeaturesDense &ref, SpectralFeaturesDense &query)
{
    ComputeLaplacianFC(ref);
    ComputeLaplacianFC(query);

    ComputeEigensDense(ref);
    ComputeEigensDense(query);

    VisualizeHistograms(ref, query);
}

void is_sym(SpectralFeaturesSparse &ref)
{
    for ( auto const &val : ref.laplacian_map )
    {
        for ( int i = 0; i < val.second.n_rows; i++ )
        {
            for ( int j = 0; j < val.second.n_cols; j++ )
            {
                if ( val.second(i, j) != val.second(j, i))
                {
                    std::cout << "NO MATCH" << std::endl;
                    std::cout << "i is: " << i << std::endl;
                    std::cout << "j  is: " << j << std::endl;
                    std::cout << "lap(i, j)  is: " << val.second(i,j) << std::endl;
                    std::cout << "lap(j, i)  is: " << val.second(j,i) << std::endl;
                }
            }
        }
    }
}

void PruneEigenvalues(SpectralFeaturesSparse &ref_sf, SpectralFeaturesSparse &query_sf)
{
    for( auto const kv : ref_sf.obj_details_map)
    {
        std::string key = kv.first;
        if (query_sf.eigval_map.find(key) != query_sf.eigval_map.end())
        {
            // First find the min sized laplacian
            int ref_size = ref_sf.eigval_map[key].n_rows;
            int query_size = query_sf.eigval_map[key].n_rows;

            // If laplacians are the same size, no pruning necessary
            if (ref_size == query_size) continue;

            arma::vec eigvals;
            bool ref_flag = false;
            bool query_flag = false;
            if (ref_size > query_size)
            {
                eigvals = ref_sf.eigval_map[key];
                ref_flag = true;
            }
            else
            {
                eigvals = query_sf.eigval_map[key];
                query_flag = true;
            }

            std::vector<double> eigs = arma::conv_to< std::vector<double> >::from(eigvals);

            std::mt19937 generator(std::random_device{}());

            int diff = abs(ref_size - query_size);
            for (int i = 0; i < diff; i++)
            {
                std::uniform_int_distribution<int> distribution(0, eigs.size() - 1);
                int rand_idx = distribution(generator);
                eigs.erase(eigs.begin() + rand_idx);
            }


            if (ref_flag)
            {
                ref_sf.eigval_map[key].reset();
                ref_sf.eigval_map[key] = arma::vec(eigs);
            }
            else
            {
                query_sf.eigval_map[key].reset();
                query_sf.eigval_map[key] = arma::vec(eigs);
            }
        }
    }
}


void MSE(SpectralFeaturesSparse &ref_sf, SpectralFeaturesSparse &query_sf)
{
    for( auto const kv : ref_sf.obj_details_map)
    {
        std::string key = kv.first;
        if (query_sf.eigval_map.find(key) != query_sf.eigval_map.end())
        {

            auto squareError = [](double a, double b) {
                double e = a-b;
                return e*e;
            };

            double sumSquared = 0;
            int size = ref_sf.eigval_map[key].size();
            for (int i = 0; i < size; i++)
            {
                sumSquared += squareError(ref_sf.eigval_map[key][i], query_sf.eigval_map[key][i]);
            }

            double mse = sumSquared / size;
            std::cout << "MSE for: " << kv.second["label"] << " = " << mse << std::endl;
        }
    }
}

void SaveEigenvalues(std::unordered_map<std::string, SpectralFeaturesSparse> &reference_map, std::unordered_map<std::string, SpectralFeaturesSparse> &query_map)
{
    json j;
    std::ofstream o("/home/nate/Development/SGPR/data/eigenvalues_1.json");

    // {
    //   reference_scans: [
    //     scan_id:
    //     ply_color:
    //     {
    //       label:
    //       global_id:
    //       eigenvalues:
    //     }
    //   ],
    //   query_scans: [
    //     scan_id:
    //     reference_scan_id:
    //     ply_color:
    //     {
    //       label:
    //       global_id:
    //       eigenvalues:
    //     }
    //   ],
    // }

    // Create reference_scans field
    std::vector<json> reference_scans;
    for ( auto &kv : reference_map )
    {
        std::string scan_id = kv.first;
        json reference_scan;
        reference_scan["scan_id"] = scan_id;
        for(auto &obj : kv.second.obj_details_map)
        {
            reference_scan[obj.first]["label"] = obj.second["label"];
            reference_scan[obj.first]["global_id"] = obj.second["global_id"];
            reference_scan[obj.first]["eigenvalues"] = kv.second.eigval_map[obj.first];
        }
        reference_scans.push_back(reference_scan);
    }

    j["reference_scans"] = reference_scans;

    // Create query_scans field
    std::vector<json> query_scans;
    for ( auto &kv : query_map )
    {
        std::string scan_id = kv.first;
        json query_scan;
        query_scan["scan_id"] = scan_id;
        query_scan["reference_scan_id"] = kv.second.referece_id;
        for(auto &obj : kv.second.obj_details_map)
        {
            query_scan[obj.first]["label"] = obj.second["label"];
            query_scan[obj.first]["global_id"] = obj.second["global_id"];
            query_scan[obj.first]["eigenvalues"] = kv.second.eigval_map[obj.first];
        }
        query_scans.push_back(query_scan);
    }

    j["query_scans"] = query_scans;

    //for ( auto const &kv : ref.obj_details_map )
    //{
    //    std::string key = kv.first;
    //    if (query.eigval_map.find(key) != query.eigval_map.end())
    //    {
    //        j[key]["label"] = kv.second["label"];
    //        j[key]["reference"] = ref.eigval_map[key];
    //        j[key]["query"] = query.eigval_map[key];
    //    }
    //}

    o << std::setw(4) << j << std::endl;
}

void NearestNeighbor(SpectralFeaturesSparse &ref, SpectralFeaturesSparse &query)
{
    //ComputeLaplacianKnn(ref);
    //ComputeLaplacianKnn(query);
    ComputeLaplacian(ref, query);

    is_sym(ref);

    ComputeEigensSparse(ref);
    ComputeEigensSparse(query);

    //PruneEigenvalues(ref, query);

    //MSE(ref, query);
    //RMSE(ref, query);

    //SaveEigenvalues(ref, query);

    //VisualizeHistograms(ref, query);
}

template<typename T>
bool swap_if_gt(T& a, T& b) {
  if (a > b) {
    std::swap(a, b);
    return true;
  }
  return false;
}

void GetGFAFeature(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud, std::vector<double> &eigenvalue_feature, pcl::PointXYZRGB &centroid) {

  // Find the variances.
  const size_t kNPoints = cloud->points.size();
  pcl::PointCloud<pcl::PointXYZRGB> variances;
  for (size_t i = 0u; i < kNPoints; ++i) {
    variances.push_back(pcl::PointXYZRGB());
    variances.points[i].x = cloud->points[i].x - centroid.x;
    variances.points[i].y = cloud->points[i].y - centroid.y;
    variances.points[i].z = cloud->points[i].z - centroid.z;
  }

  // Find the covariance matrix. Since it is symmetric, we only bother with the upper diagonal.
  const std::vector<size_t> row_indices_to_access = {0,0,0,1,1,2};
  const std::vector<size_t> col_indices_to_access = {0,1,2,1,2,2};
  Eigen::Matrix3f covariance_matrix;
  for (size_t i = 0u; i < row_indices_to_access.size(); ++i) {
    const size_t row = row_indices_to_access[i];
    const size_t col = col_indices_to_access[i];
    double covariance = 0;
    for (size_t k = 0u; k < kNPoints; ++k) {
      covariance += variances.points[k].data[row] * variances.points[k].data[col];
    }
    covariance /= kNPoints;
    covariance_matrix(row,col) = covariance;
    covariance_matrix(col,row) = covariance;
  }

  // Compute eigenvalues of covariance matrix.
  constexpr bool compute_eigenvectors = false;
  Eigen::EigenSolver<Eigen::Matrix3f> eigenvalues_solver(covariance_matrix, compute_eigenvectors);
  std::vector<float> eigenvalues(3, 0.0);
  eigenvalues.at(0) = eigenvalues_solver.eigenvalues()[0].real();
  eigenvalues.at(1) = eigenvalues_solver.eigenvalues()[1].real();
  eigenvalues.at(2) = eigenvalues_solver.eigenvalues()[2].real();
  if (eigenvalues_solver.eigenvalues()[0].imag() != 0.0 ||
      eigenvalues_solver.eigenvalues()[1].imag() != 0.0 ||
      eigenvalues_solver.eigenvalues()[2].imag() != 0.0 )
  {
      std::cout << "Eigenvalues should not have non-zero imaginary component." << std::endl;
      exit(0);
  }

  // Sort eigenvalues from smallest to largest.
  swap_if_gt(eigenvalues.at(0), eigenvalues.at(1));
  swap_if_gt(eigenvalues.at(0), eigenvalues.at(2));
  swap_if_gt(eigenvalues.at(1), eigenvalues.at(2));

  // Normalize eigenvalues.
  double sum_eigenvalues = eigenvalues.at(0) + eigenvalues.at(1) + eigenvalues.at(2);
  double e1 = eigenvalues.at(0) / sum_eigenvalues;
  double e2 = eigenvalues.at(1) / sum_eigenvalues;
  double e3 = eigenvalues.at(2) / sum_eigenvalues;
  if (e1 == e2 || e2 == e3 || e1 == e3)
  {
      std::cout << "Eigenvalues should not be equal." << std::endl;
      exit(0);
  }

  // Store inside features.
  const double sum_of_eigenvalues = e1 + e2 + e3;
  constexpr double kOneThird = 1.0/3.0;
  if(e1 == 0.0)
  {
      std::cout << "e1 should not be zero" << std::endl;
      exit(0);
  }
  if(sum_eigenvalues == 0.0)
  {
      std::cout << "sum of eigenvalues should not be 0.0" << std::endl;
      exit(0);
  }

  const double kNormalizationPercentile = 1.0;

  const double kLinearityMax = 28890.9 * kNormalizationPercentile;
  const double kPlanarityMax = 95919.2 * kNormalizationPercentile;
  const double kScatteringMax = 124811 * kNormalizationPercentile;
  const double kOmnivarianceMax = 0.278636 * kNormalizationPercentile;
  const double kAnisotropyMax = 124810 * kNormalizationPercentile;
  const double kEigenEntropyMax = 0.956129 * kNormalizationPercentile;
  const double kChangeOfCurvatureMax = 0.99702 * kNormalizationPercentile;

  const double kNPointsMax = 13200 * kNormalizationPercentile;

  eigenvalue_feature.push_back((e1 - e2) / e1 / kLinearityMax);
  eigenvalue_feature.push_back((e2 - e3) / e1 / kPlanarityMax);
  eigenvalue_feature.push_back(e3 / e1 / kScatteringMax);
  eigenvalue_feature.push_back(std::pow(e1 * e2 * e3, kOneThird) / kOmnivarianceMax);
  eigenvalue_feature.push_back((e1 - e3) / e1 / kAnisotropyMax);
  eigenvalue_feature.push_back((e1 * std::log(e1)) + (e2 * std::log(e2)) + (e3 * std::log(e3)) / kEigenEntropyMax);
  eigenvalue_feature.push_back(e3 / sum_of_eigenvalues / kChangeOfCurvatureMax);

  pcl::PointXYZRGB point_min, point_max;

  pcl::getMinMax3D(*cloud, point_min, point_max);

  double diff_x, diff_y, diff_z;

  diff_x = point_max.x - point_min.x;
  diff_y = point_max.y - point_min.y;
  diff_z = point_max.z - point_min.z;

  if (diff_z < diff_x && diff_z < diff_y) {
    eigenvalue_feature.push_back(0.2);
  } else {
    eigenvalue_feature.push_back(0.0);
  }

  // eigenvalue_feature.push_back(FeatureValue("n_points", kNPoints / kNPointsMax));

  if(eigenvalue_feature.size() != 8)
  {
      std::cout << "ERROR eigenvalue_feature vector not of size 8" << std::endl;
      exit(0);
  }

  // Check that there were no overflows, underflows, or invalid float operations.
  if (std::fetestexcept(FE_OVERFLOW)) {

      std::cout << "Overflow error in eigenvalue feature computation." << std::endl;
      exit(0);
  } else if (std::fetestexcept(FE_UNDERFLOW)) {
      std::cout << "Underflow error in eigenvalue feature computation." << std::endl;
      exit(0);
  } else if (std::fetestexcept(FE_INVALID)) {
      std::cout << "Invalid Flag error in eigenvalue feature computation." << std::endl;
      exit(0);
  } else if (std::fetestexcept(FE_DIVBYZERO)) {
      std::cout << "Divide by zero error in eigenvalue feature computation." << std::endl;;
      exit(0);
  }
}

void ComputeCentroid(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud, pcl::PointXYZRGB &centroid)
{
    pcl::CentroidPoint<pcl::PointXYZRGB> center;
    for(int i = 0; i < cloud->size(); i++)
    {
        center.add(cloud->points[i]);
    }

    center.get(centroid);
}

void SaveGFAFeatures(std::unordered_map<std::string, SpectralFeaturesSparse> &reference_map, std::unordered_map<std::string, SpectralFeaturesSparse> &query_map)
{

    json j;
    std::ofstream o("/home/nate/Development/SGPR/data/gfa_features.json");

    // {
    //   reference_scans: [
    //     scan_id:
    //     ply_color:
    //     {
    //       label:
    //       global_id:
    //       gfa_features:
    //     }
    //   ],
    //   query_scans: [
    //     scan_id:
    //     reference_scan_id:
    //     ply_color:
    //     {
    //       label:
    //       global_id:
    //       gfa_features:
    //     }
    //   ],
    // }

    // For each reference scan
    std::vector<json> reference_scans;
    for( auto &kv : reference_map )
    {
        std::string scan_id = kv.first;
        json reference_scan;
        reference_scan["scan_id"] = scan_id;
        // Loop through the obj_cloud_map
        for ( auto &cloud_kv : kv.second.obj_cloud_map )
        {
            if(cloud_kv.second->points.size() <= 3) continue;

            // Get the centroid
            pcl::PointXYZRGB centroid;
            ComputeCentroid(cloud_kv.second, centroid);

            // Get the GFA features
            std::vector<double> eigenvalue_feature;
            GetGFAFeature(cloud_kv.second, eigenvalue_feature, centroid);

            reference_scan[cloud_kv.first]["label"] = kv.second.obj_details_map[cloud_kv.first]["label"];
            reference_scan[cloud_kv.first]["global_id"] = kv.second.obj_details_map[cloud_kv.first]["global_id"];
            reference_scan[cloud_kv.first]["gfa_features"] = eigenvalue_feature;
        }

        reference_scans.push_back(reference_scan);
    }

    j["reference_scans"] = reference_scans;

    // For each query scan
    std::vector<json> query_scans;
    for( auto &kv : query_map )
    {
        std::string scan_id = kv.first;
        json query_scan;
        query_scan["scan_id"] = scan_id;
        query_scan["reference_scan_id"] = kv.second.referece_id;

        // Loop through the obj_cloud_map
        for ( auto &cloud_kv : kv.second.obj_cloud_map )
        {
            if(cloud_kv.second->points.size() <= 3) continue;

            // Get the centroid
            pcl::PointXYZRGB centroid;
            ComputeCentroid(cloud_kv.second, centroid);

            // Get the GFA features
            std::vector<double> eigenvalue_feature;
            GetGFAFeature(cloud_kv.second, eigenvalue_feature, centroid);

            query_scan[cloud_kv.first]["label"] = kv.second.obj_details_map[cloud_kv.first]["label"];
            query_scan[cloud_kv.first]["global_id"] = kv.second.obj_details_map[cloud_kv.first]["global_id"];
            query_scan[cloud_kv.first]["gfa_features"] = eigenvalue_feature;
        }

        query_scans.push_back(query_scan);
    }

    j["query_scans"] = query_scans;
    o << std::setw(4) << j << std::endl;
}

int main()
{
    //std::string testConfig= "/home/nate/Development/SGPR/config/test.json";

    //std::ifstream testFile(testConfig);

    //json testData = json::parse(testFile);

    // Key of the reference_map and query_map is their corresponding scan id's
    std::unordered_map<std::string, SpectralFeaturesSparse> reference_map;
    std::unordered_map<std::string, SpectralFeaturesSparse> query_map;

    ParseConfig(reference_map, query_map);

    for(auto &kv : reference_map)
    {
        PopulateSemanticEigenMap(kv.second);
    }

    for(auto &kv : query_map)
    {
        PopulateSemanticEigenMap(kv.second);
    }

    SaveGFAFeatures(reference_map, query_map);

    // NOTE: This is for spectral eigenvalue methods
    //for(auto &ref_kv : reference_map)
    //{
    //    for(auto &query_kv : query_map)
    //    {
    //        if (query_kv.second.referece_id != ref_kv.first) continue;
    //        NearestNeighbor(ref_kv.second, query_kv.second);
    //    }
    //}

    //SaveEigenvalues(reference_map, query_map);

    return 1;
}
