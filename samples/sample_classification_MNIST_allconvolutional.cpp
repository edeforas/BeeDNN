// all convolutional MNIST classification with a conv2d (no poolmax layers)
// accuracy > 99% after 20 epochs, 40s/epochs
// conv2d speed is not fully optimized yet.

#include <iostream>
#include <chrono>

#include "Net.h"
#include "NetTrain.h"
#include "MNISTReader.h"
#include "Metrics.h"

#include "LayerActivation.h"
#include "LayerConvolution2D.h"
#include "LayerChannelBias.h"
#include "LayerDense.h"
#include "LayerDropout.h"
#include "LayerSoftmax.h"

using namespace std;
using namespace beednn;

NetTrain netTrain;
int iEpoch;
chrono::steady_clock::time_point start;

//////////////////////////////////////////////////////////////////////////////
void epoch_callback()
{
	//compute epoch time
	chrono::steady_clock::time_point next = chrono::steady_clock::now();
	auto delta = chrono::duration_cast<std::chrono::milliseconds>(next - start).count();
	start = next;

    iEpoch++;
    cout << "Epoch: " << iEpoch << " duration: " << delta << " ms" << endl;
	cout << "TrainLoss: " << netTrain.get_current_train_loss() << " TrainAccuracy: " << netTrain.get_current_train_accuracy() << " %" ;
	cout << " ValidationAccuracy: " << netTrain.get_current_validation_accuracy() << " %" << endl;

	cout << endl;
}
//////////////////////////////////////////////////////////////////////////////
int main()
{
    iEpoch = 0;

	cout << "all convolutional MNIST classification with a conv2d (no poolmax layers)" << endl;
	cout << " accuracy> 99% after 20 epochs, 25s/epochs" << endl;
	cout << "conv2d speed is not fully optimized yet" << endl;

	//load and normalize MNIST data
    cout << "Loading MNIST database..." << endl;
    MNISTReader mr;
    if(!mr.load("."))
    {
        cout << "MNIST samples not found, please check the *.ubyte files are in the executable folder" << endl;
        return -1;
    }

	//create a all convolutional net:
	Net model;
	model.add(new LayerConvolution2D(28, 28, 1, 3, 3, 8));
	model.add(new LayerChannelBias(26,26,8));
	model.add(new LayerActivation("Relu"));

	model.add(new LayerConvolution2D(26, 26, 8, 3, 3, 8, 2, 2));
	model.add(new LayerChannelBias(12,12,8));
	model.add(new LayerActivation("Relu"));
	model.add(new LayerDropout(0.3f));

	model.add(new LayerConvolution2D(12, 12, 8, 3, 3, 16));
	model.add(new LayerChannelBias(10,10,16));
	model.add(new LayerActivation("Relu"));
	model.add(new LayerDropout(0.3f));

	model.add(new LayerDense(10 * 10 * 16, 256));

	model.add(new LayerActivation("Relu"));
	model.add(new LayerDense(256, 10));
	model.add(new LayerSoftmax());

	//set train options
	netTrain.set_epochs(20);
	netTrain.set_batchsize(32);
	netTrain.set_loss("SparseCategoricalCrossEntropy");
	netTrain.set_epoch_callback(epoch_callback); //optional, show progress
	netTrain.set_train_data(mr.train_data(), mr.train_truth());
	netTrain.set_validation_data(mr.validation_data(), mr.validation_truth()); //optional, not used for training, helps to keep the final best model

	// train net
	cout << "Training..." << endl << endl;
	start = chrono::steady_clock::now();
	netTrain.fit(model);

	// show train results
	MatrixFloat mClassPredicted;
	model.predict_classes(mr.train_data(), mClassPredicted);
	Metrics metricsTrain;
	metricsTrain.compute(mr.train_truth(), mClassPredicted);
	cout << "Ref accuracy: " << metricsTrain.accuracy() << " %" << endl;

	MatrixFloat mClassVal;
	model.predict_classes(mr.validation_data(), mClassVal);
	Metrics metricsVal;
	metricsVal.compute(mr.validation_truth(), mClassVal);
	cout << "Val accuracy: " << metricsVal.accuracy() << " %" << endl;
	cout << "Val confusion matrix:" << endl << toString(metricsVal.confusion_matrix()) << endl;
	
	//testu function
	if (metricsVal.accuracy() < 99.f)
	{
		cout << "Test failed! val accuracy=" << metricsVal.accuracy() << endl;
		return -1;
	}

	cout << "Test succeded." << endl;
    return 0;
}
