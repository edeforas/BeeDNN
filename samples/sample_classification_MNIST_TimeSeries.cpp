// simple MNIST classification using time series, all image seen row by row
// validation accuracy > 97%, after 30 epochs (0.2s by epochs)

#include <iostream>
#include <chrono>
using namespace std;

#include "Net.h"
#include "NetTrain.h"
#include "MNISTReader.h"
#include "Metrics.h"

#include "LayerActivation.h"
#include "LayerDense.h"
#include "LayerTimeDistributedDense.h"
#include "LayerSimplestRNN.h"
#include "LayerDropout.h"
#include "LayerSoftmax.h"

Net model;
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
	cout << "simple MNIST classification using time series, all image seen row by row" << endl;
	cout << "validation accuracy > 97%, after 30 epochs (0.2s by epochs)" << endl;

    iEpoch = 0;

	//load and normalize MNIST data
    cout << "Loading MNIST database..." << endl;
    MNISTReader mr;
    if(!mr.load("."))
    {
        cout << "MNIST samples not found, please check the *.ubyte files are in the executable folder" << endl;
        return -1;
    }
  
	//create simple net:
	model.add(new LayerTimeDistributedDense(28,8));
//	net.add(new LayerSimplestRNN(28)); // WIP, removed for now
	model.add(new LayerDense(28*8, 32));
	model.add(new LayerActivation("Relu"));
	model.add(new LayerDropout(0.2f)); //reduce overfitting
	model.add(new LayerDense(32, 10));
	model.add(new LayerSoftmax());

	//setup train options
	netTrain.set_epochs(30);
	netTrain.set_batchsize(128);
	netTrain.set_loss("SparseCategoricalCrossEntropy");
	netTrain.set_epoch_callback(epoch_callback); //optional, to show the progress
	netTrain.set_train_data(mr.train_data(),mr.train_truth());
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
	cout << "Train accuracy: " << metricsTrain.accuracy() << " %" << endl;

	MatrixFloat mClassVal;
	model.predict_classes(mr.validation_data(), mClassVal);
	Metrics metricsVal;
	metricsVal.compute(mr.validation_truth(), mClassVal);
	cout << "Validation accuracy: " << metricsVal.accuracy() << " %" << endl;
	cout << "Validation confusion matrix:" << endl << toString(metricsVal.confusion_matrix()) << endl;

	//testu function
	if (metricsVal.accuracy() < 97.f)
	{
		cout << "Test failed! val accuracy=" << metricsVal.accuracy() << endl;
		return -1;
	}

	cout << "Test succeded." << endl;
    return 0;
}
