cmake_minimum_required(VERSION 3.20.0)
project (BeeDNN)
enable_testing()

# if eigen is installed, use it to speed up training
if(DEFINED ENV{EIGEN_PATH})
    message(STATUS "Using Eigen at EIGEN_PATH env")
    include_directories($ENV{EIGEN_PATH})
    add_compile_definitions(USE_EIGEN)
else()
    message(STATUS "Not using Eigen (EIGEN_PATH env not defined)")
    message(STATUS "Training will be slow.")
    message(STATUS "To speed up, install Eigen headers from https://eigen.tuxfamily.org/")
endif()

add_subdirectory(src)
add_subdirectory(tests)
add_subdirectory(samples)
