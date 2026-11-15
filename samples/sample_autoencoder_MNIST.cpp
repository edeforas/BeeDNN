#include <chrono>
#include <iostream>

#include "Net.h"
#include "NetTrain.h"
#include "MNISTReader.h"

#include "LayerActivation.h"
#include "LayerChannelBias.h"
#include "LayerConvolution2D.h"
#include "LayerDense.h"
#include "LayerMaxPool2D.h"
#include "LayerUpSampling2D.h"
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
		<< " Train loss: " << netTrain.get_current_train_loss()
		<< " Validation loss: " << netTrain.get_current_validation_loss() << endl;
}

int main()
{
	cout << "Convolutional MNIST autoencoder" << endl;
	cout << "Loading MNIST database..." << endl;
	MNISTReader mnist;
	if (!mnist.load("."))
	{
		cout << "MNIST samples not found, please check the *.ubyte files are in the executable folder" << endl;
		return -1;
	}

	Net model;
	model.set_classification_mode(false);

	// Encode 28x28 images to a 32-value latent representation.
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

	model.add(new LayerDense(7 * 7 * 16, 32));
	model.add(new LayerActivation("Relu"));

	// Decode the latent vector back to a 28x28 image with convolutions.
	model.add(new LayerDense(32, 7 * 7 * 16));
	model.add(new LayerActivation("Relu"));
	model.add(new LayerUpSampling2D(7, 7, 16));
	model.add(new LayerUpSampling2D(14, 14, 16));
	model.add(new LayerZeroPadding2D(28, 28, 16, 1));
	model.add(new LayerConvolution2D(30, 30, 16, 3, 3, 16));
	model.add(new LayerChannelBias(28, 28, 16));
	model.add(new LayerActivation("Relu"));
	model.add(new LayerConvolution2D(28, 28, 16, 1, 1, 1));
	model.add(new LayerChannelBias(28, 28, 1));
	model.add(new LayerActivation("Sigmoid"));

	netTrain.set_epochs(10);
	netTrain.set_batchsize(64);
	netTrain.set_loss("MeanSquaredError");
	netTrain.set_epoch_callback(epoch_callback);
	netTrain.set_train_data(mnist.train_data(), mnist.train_data());
	netTrain.set_validation_data(mnist.validation_data(), mnist.validation_data());

	cout << "Training..." << endl << endl;
	start = chrono::steady_clock::now();
	netTrain.fit(model);

	MatrixFloat validationReconstruction;
	model.predict(mnist.validation_data(), validationReconstruction);
	const float validationLoss = (validationReconstruction - mnist.validation_data()).cwiseAbs2().mean();
	cout << "Validation reconstruction MSE: " << validationLoss << endl;
	return 0;
}
