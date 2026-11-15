#include <chrono>
#include <iostream>

#include "Net.h"
#include "NetTrain.h"
#include "MNISTReader.h"
#include "Metrics.h"

#include "LayerActivation.h"
#include "LayerChannelBias.h"
#include "LayerConvolution2D.h"
#include "LayerDense.h"
#include "LayerDropout.h"
#include "LayerMaxPool2D.h"
#include "LayerRandomAffine2D.h"
#include "LayerSoftmax.h"
#include "LayerZeroPadding2D.h"

using namespace std;
using namespace beednn;

NetTrain netTrain;
int iEpoch;
chrono::steady_clock::time_point start;

void epoch_callback()
{
	const chrono::steady_clock::time_point next = chrono::steady_clock::now();
	const auto duration = chrono::duration_cast<chrono::milliseconds>(next - start).count();
	start = next;

	iEpoch++;
	cout << "Epoch: " << iEpoch << " duration: " << duration << " ms"
		<< " TrainLoss: " << netTrain.get_current_train_loss()
		<< " TrainAccuracy: " << netTrain.get_current_train_accuracy() << " %"
		<< " ValidationAccuracy: " << netTrain.get_current_validation_accuracy() << " %" << endl;
}

int main()
{
	cout << "MNIST Conv2D classification with random rotation and translation augmentation" << endl;
	cout << "Loading MNIST database..." << endl;
	MNISTReader mnist;
	if (!mnist.load("."))
	{
		cout << "MNIST samples not found, please check the *.ubyte files are in the executable folder" << endl;
		return -1;
	}

	Net model;
	model.add(new LayerRandomAffine2D(28, 28, 1, 10.f, 2.f, 2.f));

	model.add(new LayerZeroPadding2D(28, 28, 1, 1));
	model.add(new LayerConvolution2D(30, 30, 1, 3, 3, 8));
	model.add(new LayerChannelBias(28, 28, 8));
	model.add(new LayerActivation("Relu"));
	model.add(new LayerMaxPool2D(28, 28, 8));

	model.add(new LayerZeroPadding2D(14, 14, 8, 1));
	model.add(new LayerConvolution2D(16, 16, 8, 3, 3, 16));
	model.add(new LayerChannelBias(14, 14, 16));
	model.add(new LayerActivation("Relu"));
	model.add(new LayerMaxPool2D(14, 14, 16));

	model.add(new LayerZeroPadding2D(7, 7, 16, 1));
	model.add(new LayerConvolution2D(9, 9, 16, 3, 3, 24));
	model.add(new LayerChannelBias(7, 7, 24));
	model.add(new LayerActivation("Relu"));
	model.add(new LayerDense(7 * 7 * 24, 128));
	model.add(new LayerActivation("Relu"));
	model.add(new LayerDropout(0.3f));
	model.add(new LayerDense(128, 10));
	model.add(new LayerSoftmax());

	netTrain.set_epochs(10);
	netTrain.set_batchsize(64);
	netTrain.set_loss("SparseCategoricalCrossEntropy");
	netTrain.set_epoch_callback(epoch_callback);
	netTrain.set_train_data(mnist.train_data(), mnist.train_truth());
	netTrain.set_validation_data(mnist.validation_data(), mnist.validation_truth());

	cout << "Training..." << endl << endl;
	iEpoch = 0;
	start = chrono::steady_clock::now();
	netTrain.fit(model);

	MatrixFloat predicted;
	model.predict_classes(mnist.validation_data(), predicted);
	Metrics metrics;
	metrics.compute(mnist.validation_truth(), predicted);
	cout << "Validation accuracy: " << metrics.accuracy() << " %" << endl;
	cout << "Validation confusion matrix:" << endl << toString(metrics.confusion_matrix()) << endl;
	return 0;
}
