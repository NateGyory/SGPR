#include <pcl/point_cloud.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>

int main(int argc, char *argv[]) {
  // Create point cloud 5x5
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud1(new pcl::PointCloud<pcl::PointXYZ>());
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud2(new pcl::PointCloud<pcl::PointXYZ>());

  pcl::PointXYZ pt;
  for (int x = 0; x < 5; x++) {
    pt.x = x;
    for (int y = 0; y < 5; y++) {
      pt.y = y;
      for (int z = 0; z < 5; z++) {
        pt.z = z;
        cloud1->points.push_back(pt);
      }
    }
  }

  for (int x = 0; x < 10; x++) {
    pt.x = double(x) / 2;
    for (int y = 0; y < 10; y++) {
      pt.y = double(y) / 2;
      for (int z = 0; z < 10; z++) {
        pt.z = double(z) / 2;
        cloud2->points.push_back(pt);
      }
    }
  }

  pcl::visualization::PCLVisualizer viewer("Viz");
  viewer.setBackgroundColor(0.0f, 0.0f, 0.0f);
  viewer.addPointCloud(cloud1, "cloud1");
  viewer.setPointCloudRenderingProperties(
      pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud1");

  pcl::visualization::PCLVisualizer viewer2("Viz");
  viewer2.setBackgroundColor(0.0f, 0.0f, 0.0f);
  viewer2.addPointCloud(cloud2, "cloud2");
  viewer2.setPointCloudRenderingProperties(
      pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud2");

  while (!viewer.wasStopped() || !viewer2.wasStopped()) {
    viewer.spinOnce();
    viewer2.spinOnce();
  }

  return 0;
}
