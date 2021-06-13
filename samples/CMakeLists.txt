cmake_minimum_required(VERSION 3.0.0)

if (UNIX)
	list(APPEND CMAKE_CXX_FLAGS "-pthread -std=c++11 ${CMAKE_CXX_FLAGS}")
endif (UNIX)

project (test_samples)

include_directories(../src)
include_directories($ENV{EIGEN_PATH})
add_compile_definitions("USE_EIGEN")

set(CPP_files
../src/CsvFileReader.cpp
../src/Activations.cpp
../src/ConfusionMatrix.cpp
../src/DataSource.cpp
../src/KMeans.cpp
../src/KMeansTrain.cpp
../src/Layer.cpp
../src/LayerAffine.cpp
../src/LayerBias.cpp
../src/LayerActivation.cpp
../src/LayerChannelBias.cpp
../src/LayerConvolution2D.cpp
../src/LayerDense.cpp
../src/LayerGain.cpp
../src/LayerGlobalAffine.cpp
../src/LayerGlobalBias.cpp
../src/LayerGlobalGain.cpp
../src/LayerGaussianNoise.cpp
../src/LayerDropout.cpp
../src/LayerFactory.cpp
../src/LayerPRelu.cpp
../src/LayerPoolMax2D.cpp
../src/LayerRRelu.cpp
../src/LayerSoftmax.cpp
../src/LayerSoftmin.cpp
../src/LayerSimpleRNN.cpp
../src/LayerUniformNoise.cpp
../src/Loss.cpp
../src/Matrix.cpp
../src/MNISTReader.cpp
../src/CIFAR10Reader.cpp
../src/MetaOptimizer.cpp
../src/Net.cpp
../src/NetTrain.cpp
../src/NetUtil.cpp
../src/Optimizer.cpp
../src/Regularizer.cpp
../src/TimeSeriesUtil.cpp
)

set(H_files
../src/CsvFileReader.h
../src/Activations.h
../src/ConfusionMatrix.h
../src/DataSource.h
../src/KMeans.h
../src/KMeansTrain.h
../src/Layer.h
../src/LayerActivation.h
../src/LayerAffine.h
../src/LayerBias.h
../src/LayerChannelBias.h
../src/LayerConvolution2D.h
../src/LayerDense.h
../src/LayerGlobalAffine.cpp
../src/LayerGlobalBias.cpp
../src/LayerGlobalGain.cpp
../src/LayerGaussianNoise.h
../src/LayerGain.h
../src/LayerDropout.h
../src/LayerFactory.h
../src/LayerPRelu.h
../src/LayerPoolMax2D.h
../src/LayerRRelu.h
../src/LayerSoftmax.h
../src/LayerSoftmin.h
../src/LayerSimpleRNN.h
../src/LayerUniformNoise.h
../src/Loss.h
../src/Matrix.h
../src/MNISTReader.h
../src/CIFAR10Reader.h
../src/MetaOptimizer.h
../src/Net.h
../src/NetTrain.h
../src/NetUtil.h
../src/Optimizer.h
../src/Regularizer.h
../src/TimeSeriesUtil.h
)

add_executable(sample_classification_xor
	sample_classification_xor.cpp
	${CPP_files}
	${H_files}
)

add_executable(sample_classification_MNIST
	sample_classification_MNIST.cpp
	${CPP_files}
	${H_files}
)

add_executable(sample_classification_MNIST_allconvolutional
	sample_classification_MNIST_allconvolutional.cpp
	${CPP_files}
	${H_files}
)

add_executable(sample_classification_MNIST_poolmax2d
	sample_classification_MNIST_poolmax2d.cpp
	${CPP_files}
	${H_files}
)

add_executable(sample_regression_sin
	sample_regression_sin.cpp
	${CPP_files}
	${H_files}
)

add_executable(sample_sunspot_prediction
	sample_sunspot_prediction.cpp
	${CPP_files}
	${H_files}
)

add_executable(sample_KMeans_MNIST
	sample_KMeans_MNIST.cpp
	${CPP_files}
	${H_files}
)

add_executable(sample_MetaOptimizer_MNIST
	sample_MetaOptimizer_MNIST.cpp
	${CPP_files}
	${H_files}
)
 
add_executable(sample_classification_CIFAR10
	sample_classification_CIFAR10.cpp
	${CPP_files}
	${H_files}
)
