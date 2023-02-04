cmake_minimum_required(VERSION 3.16)
project(SGPR)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_BUILD_TYPE Debug)

#find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)
find_package(Armadillo REQUIRED)
find_package(nlohmann_json 3.2.0 REQUIRED)
#find_package(Matplot++ REQUIRED)
find_package(Spectra PATHS /home/nate/Build/spectra/install NO_DEFAULT_PATH)

#find_package(Open3D PATHS /home/nate/Build/Open3D/install NO_DEFAULT_PATH)
#target_link_libraries(SGPR PRIVATE ${EIGEN3_LIBRARIES} Open3D::Open3D)


#add_definitions(${PCL_DEFINITIONS})

file(GLOB_RECURSE SRCS CONFIGURE_DEPENDS "src/main.cpp")
#file(GLOB_RECURSE PROOFS_SRCS CONFIGURE_DEPENDS "src/*.cpp")
#file(GLOB_RECURSE PROOFS_SRCS CONFIGURE_DEPENDS "src/spectralProof.cpp")

#add_executable(SGPR ${SRCS})
#target_link_libraries(SGPR PRIVATE ${PCL_LIBRARIES} ${ARMADILLO_LIBRARIES} nlohmann_json::nlohmann_json Matplot++::matplot)
#target_include_directories(SGPR PRIVATE ${PCL_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/include ${ARMADILLO_INCLUDE_DIRS})

add_executable(normals src/surface_normals.cpp)
target_link_libraries(normals PRIVATE ${PCL_LIBRARIES} ${ARMADILLO_LIBRARIES})
target_include_directories(normals PRIVATE ${PCL_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/include)

add_executable(cube src/cube.cpp)
target_link_libraries(cube PRIVATE ${PCL_LIBRARIES})
target_include_directories(cube PRIVATE ${PCL_INCLUDE_DIRS})
