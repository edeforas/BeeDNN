cmake_minimum_required(VERSION 3.0.0)

if (UNIX)
	list(APPEND CMAKE_CXX_FLAGS "-fopenmp" ${CMAKE_CXX_FLAGS})
endif (UNIX)

project (test_samples)

include_directories(../src)
include_directories($ENV{EIGEN_PATH})
add_compile_definitions("USE_EIGEN")

file(GLOB SOURCE_FILES ../src/*.cpp ../src/*.h)
 
add_executable(sample_classification_xor
	sample_classification_xor.cpp
	${SOURCE_FILES}
)

add_executable(sample_classification_MNIST
	sample_classification_MNIST.cpp
	${SOURCE_FILES}
)

add_executable(sample_classification_MNIST_allconvolutional
	sample_classification_MNIST_allconvolutional.cpp
	${SOURCE_FILES}
)

add_executable(sample_classification_MNIST_poolmax2d
	sample_classification_MNIST_poolmax2d.cpp
	${SOURCE_FILES}
)

add_executable(sample_regression_sin
	sample_regression_sin.cpp
	${SOURCE_FILES}
)

add_executable(sample_sunspot_prediction
	sample_sunspot_prediction.cpp
	${SOURCE_FILES}
)

add_executable(sample_KMeans_MNIST
	sample_KMeans_MNIST.cpp
	${SOURCE_FILES}
)

add_executable(sample_MetaOptimizer_MNIST
	sample_MetaOptimizer_MNIST.cpp
	${SOURCE_FILES}
)
 
add_executable(sample_classification_CIFAR10
	sample_classification_CIFAR10.cpp
	${SOURCE_FILES}
)
