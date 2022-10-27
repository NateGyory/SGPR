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

#include <matplot/matplot.h>

#include <armadillo>

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

void ComputeLaplacian(SpectralFeaturesSparse &sf)
{
    for( auto const kv : sf.obj_cloud_map )
    {
        pcl::KdTreeFLANN<pcl::PointXYZRGB> kdTree;
        kdTree.setInputCloud(kv.second);
        pcl::PointXYZ searchPoint;

        int rows = kv.second->points.size();
        int cols = rows;

        double radius = 0.1f;
        unsigned int max_nn = 200;
        //int neighbors = 6;
        double smallestDistance = GetSmallestDistance(kdTree, rows);
        double bias = 1.0 - smallestDistance;

        arma::sp_mat laplacian(rows, cols);

        std::cout << "Laplacian size: " << rows << std::endl;
        for (int i = 0; i < rows; i++) {
            std::vector<int> indicies_found;
            std::vector<float> squaredDistances;
            kdTree.radiusSearch(i, radius, indicies_found, squaredDistances, max_nn);
            //kdTree.nearestKSearch(i, neighbors, indicies_found, squaredDistances);

            int num_edges = indicies_found.size() - 1;
            std::cout << num_edges << std::endl;
            laplacian(i,i) = num_edges;

            for (int j = 1; j < indicies_found.size(); j++)
            {
                laplacian(i, indicies_found[j]) = -1 / (sqrt(squaredDistances[j]) + bias);
                laplacian(indicies_found[j], i) = -1 / (sqrt(squaredDistances[j]) + bias);
            }
        }

        //std::cout << laplacian << std::endl;
        sf.laplacian_map[kv.first] = laplacian;
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
        sf.laplacian_map[kv.first] = laplacian; } }

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

template<class T>
void ParseConfig(T &sf)
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

            std::string label = kv.second["label"].dump();
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

void NearestNeighbor(SpectralFeaturesSparse &ref, SpectralFeaturesSparse &query)
{
    //ComputeLaplacianKnn(ref);
    //ComputeLaplacianKnn(query);
    ComputeLaplacian(ref);
    ComputeLaplacian(query);

    is_sym(ref);

    ComputeEigensSparse(ref);
    ComputeEigensSparse(query);

    VisualizeHistograms(ref, query);
}

void StaticRadius()
{

}

void AdaptiveRadius()
{

}

int main()
{
    //SpectralFeaturesDense ref_sf;
    //SpectralFeaturesDense query_sf;
    SpectralFeaturesSparse ref_sf;
    SpectralFeaturesSparse query_sf;

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

    // TODO
    // nearest neightbor laplacian
    // adaptive threshold laplacian
    // Probability transition laplacian

    //FullyConnected(ref_sf, query_sf);
    NearestNeighbor(ref_sf, query_sf);

    return 1;
}
