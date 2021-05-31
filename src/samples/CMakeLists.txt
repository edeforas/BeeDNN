cmake_minimum_required(VERSION 3.0.0)

if (UNIX)
	list(APPEND CMAKE_CXX_FLAGS "-pthread -std=c++11 ${CMAKE_CXX_FLAGS}")
endif (UNIX)

project (test_samples)

include_directories(..)
include_directories($ENV{EIGEN_PATH})
add_compile_definitions("USE_EIGEN")

set(CPP_files
../CsvFileReader.cpp
../Activations.cpp
../ConfusionMatrix.cpp
../DataSource.cpp
../KMeans.cpp
../KMeansTrain.cpp
../Layer.cpp
../LayerAffine.cpp
../LayerBias.cpp
../LayerActivation.cpp
../LayerChannelBias.cpp
../LayerConvolution2D.cpp
../LayerDense.cpp
../LayerGain.cpp
../LayerGlobalAffine.cpp
../LayerGlobalBias.cpp
../LayerGlobalGain.cpp
../LayerGaussianNoise.cpp
../LayerDropout.cpp
../LayerFactory.cpp
../LayerPRelu.cpp
../LayerPoolMax2D.cpp
../LayerRRelu.cpp
../LayerSoftmax.cpp
../LayerSoftmin.cpp
../LayerSimpleRNN.cpp
../LayerUniformNoise.cpp
../Loss.cpp
../Matrix.cpp
../MNISTReader.cpp
../CIFAR10Reader.cpp
../MetaOptimizer.cpp
../Net.cpp
../NetTrain.cpp
../NetUtil.cpp
../Optimizer.cpp
../Regularizer.cpp
../TimeSeriesUtil.cpp
)

set(H_files
../CsvFileReader.h
../Activations.h
../ConfusionMatrix.h
../DataSource.h
../KMeans.h
../KMeansTrain.h
../Layer.h
../LayerActivation.h
../LayerAffine.h
../LayerBias.h
../LayerChannelBias.h
../LayerConvolution2D.h
../LayerDense.h
../LayerGlobalAffine.cpp
../LayerGlobalBias.cpp
../LayerGlobalGain.cpp
../LayerGaussianNoise.h
../LayerGain.h
../LayerDropout.h
../LayerFactory.h
../LayerPRelu.h
../LayerPoolMax2D.h
../LayerRRelu.h
../LayerSoftmax.h
../LayerSoftmin.h
../LayerSimpleRNN.h
../LayerUniformNoise.h
../Loss.h
../Matrix.h
../MNISTReader.h
../CIFAR10Reader.h
../MetaOptimizer.h
../Net.h
../NetTrain.h
../NetUtil.h
../Optimizer.h
../Regularizer.h
../TimeSeriesUtil.h
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
