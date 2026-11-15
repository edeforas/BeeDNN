#include <chrono>
#include <iostream>

#include "Net.h"
#include "NetTrain.h"
#include "MNISTReader.h"
#include "Metrics.h"

#include "LayerActivation.h"
#include "LayerDense.h"
#include "LayerDropout.h"
#include "LayerSoftmax.h"
#include "LayerTimeDistributedDense.h"
#include "LayerTransformerEncoder.h"

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
	cout << "MNIST classification with a Transformer encoder" << endl;
	cout << "Each image row is one token; fixed sinusoidal position encoding is added by the encoder." << endl;
	iEpoch = 0;

	cout << "Loading MNIST database..." << endl;
	MNISTReader mnist;
	if (!mnist.load("."))
	{
		cout << "MNIST samples not found, please check the *.ubyte files are in the executable folder" << endl;
		return -1;
	}

	Net model;
	model.add(new LayerTimeDistributedDense(28, 16));
	model.add(new LayerTransformerEncoder(28, 16, 32, 4));
	model.add(new LayerDense(28 * 16, 128));
	model.add(new LayerActivation("Relu"));
	model.add(new LayerDropout(0.2f));
	model.add(new LayerDense(128, 10));
	model.add(new LayerSoftmax());

	netTrain.set_epochs(10);
	netTrain.set_batchsize(64);
	netTrain.set_loss("SparseCategoricalCrossEntropy");
	netTrain.set_epoch_callback(epoch_callback);
	netTrain.set_train_data(mnist.train_data(), mnist.train_truth());
	netTrain.set_validation_data(mnist.validation_data(), mnist.validation_truth());

	cout << "Training..." << endl << endl;
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
