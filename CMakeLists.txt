cmake_minimum_required(VERSION 3.16)
project(SGPR)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_BUILD_TYPE Debug)

find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)
find_package(Armadillo REQUIRED)
#find_package(Open3D PATHS /home/nate/Build/Open3D/install NO_DEFAULT_PATH)
#target_link_libraries(SGPR PRIVATE ${EIGEN3_LIBRARIES} Open3D::Open3D)

message("PCL libs are: ${PCL_LIBRARRIES}")

#add_definitions(${PCL_DEFINITIONS})

file(GLOB_RECURSE SRCS CONFIGURE_DEPENDS "src/*.cpp")

add_executable(SGPR ${SRCS})
target_link_libraries(SGPR PRIVATE ${EIGEN3_LIBRARIES} ${PCL_LIBRARIES} ${ARMADILLO_LIBRARIES})
target_include_directories(SGPR PRIVATE ${EIGEN3_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/include ${ARMADILLO_INCLUDE_DIRS})
