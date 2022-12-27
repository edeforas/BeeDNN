// simple toy classification CIFAR10, with a convolutional Layers, small accuracy (60% after 15 epochs), 20s /epoch
// It shows and save the current model vs. epochs on disk
// To stop by anytime, type CTRL+C"

// use a DataSource for Data augmentations (WIP)

#include <chrono>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
using namespace std;

#include "Net.h"
#include "NetTrain.h"
#include "NetUtil.h"
#include "CIFAR10Reader.h"
#include "ConfusionMatrix.h"

#include "LayerActivation.h"
#include "LayerConvolution2D.h"
#include "LayerChannelBias.h"
#include "LayerDense.h"
#include "LayerDropout.h"
#include "LayerSoftmax.h"
#include "LayerMaxPool2D.h"
#include "LayerConvolution2D.h"
#include "LayerRandomFlip.h"

Net model;
NetTrain netTrain;
CIFAR10Reader ds;
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
	cout  << " TrainAccuracy: " << netTrain.get_current_train_accuracy() << " %" ;
	cout << " ValidationAccuracy: " << netTrain.get_current_validation_accuracy() << " %" << endl;

	// save solution
	ostringstream sFile;
	sFile << "model_" << "epoch_" << iEpoch << "_accuracy_" << fixed << setprecision(2) << netTrain.get_current_validation_accuracy() << ".json";
	NetUtil::save(sFile.str(), netTrain.model(), netTrain); //save train parameters and net
}
//////////////////////////////////////////////////////////////////////////////
int main()
{
	cout << "Simple toy classification CIFAR10, small accuracy (60% after 15 epochs), 20s /epoch" << endl;
	cout << "It shows and save the current model vs. epoch on disk" << endl;
	cout << "To stop by anytime, type CTRL+C" << endl;

	iEpoch = 0;

	//load and normalize CIFAR10 data
    cout << "Loading CIFAR10 database..." << endl;
    
    if(!ds.load(".")) // also divide images by 256
	{
		cout << "CIFAR10 samples not found, please check the CIFAR10 *.bin files are in the executable folder" << endl;
		return -1;
	}
  
	//create simple convolutionnal net:
	model.add(new LayerConvolution2D(32, 32, 3, 3, 3, 8));
	model.add(new LayerChannelBias(30, 30, 8));
	model.add(new LayerActivation("Relu"));
	model.add(new LayerMaxPool2D(30, 30, 8, 2, 2));
	model.add(new LayerConvolution2D(15, 15, 8, 3, 3, 16));
	model.add(new LayerChannelBias(13, 13, 16));
	model.add(new LayerActivation("Relu"));
	model.add(new LayerDense(13 * 13 * 16, 256));
	model.add(new LayerActivation("Relu"));
	model.add(new LayerDense(256, 10));
	model.add(new LayerSoftmax());

	//setup train options
	netTrain.set_epochs(15);
	netTrain.set_batchsize(256);
	netTrain.set_loss("SparseCategoricalCrossEntropy");
	netTrain.set_epoch_callback(epoch_callback); //optional, show the progress
	netTrain.set_train_data(ds.train_data(),ds.train_truth());
	netTrain.set_validation_data(ds.validation_data(), ds.validation_truth());

	// train net
	cout << "Training..." << endl << endl;
	start = chrono::steady_clock::now();
	netTrain.fit(model);

	// show train and val confusion matrix results
	MatrixFloat mClassPredicted;
	model.predict_classes(ds.train_data(), mClassPredicted);
	ConfusionMatrix cmRef;
	ClassificationResult crRef = cmRef.compute(ds.train_truth(), mClassPredicted);
	cout << "Train accuracy: " << crRef.accuracy << " %" << endl;
	cout << "Train confusion matrix:" << endl << toString(crRef.mConfMat) << endl;

	MatrixFloat mValClass;
	model.predict_classes(ds.validation_data(), mValClass);
	ConfusionMatrix cmValidation;
	ClassificationResult crValidation = cmValidation.compute(ds.validation_truth(), mValClass);
	cout << "Validation accuracy: " << crValidation.accuracy << " %" << endl;
	cout << "Validation confusion matrix:" << endl << toString(crValidation.mConfMat) << endl;

    return 0;
}
