// simple toy classification CIFAR10, with a dense Layer, low accuracy for now (50%), but simple
// 6s /epoch

#include <iostream>
#include <chrono>
using namespace std;

#include "Net.h"
#include "NetTrain.h"
#include "CIFAR10Reader.h"
#include "ConfusionMatrix.h"

#include "LayerActivation.h"
#include "LayerConvolution2D.h"
#include "LayerChannelBias.h"
#include "LayerDense.h"
#include "LayerDropout.h"
#include "LayerSoftmax.h"
#include "LayerPoolMax2D.h"

Net net;
NetTrain netTrain;
MatrixFloat mRefImages, mRefLabels, mValImages, mValLabels;
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
	cout << "Simple toy classification CIFAR10, with a dense Layer, low accuracy for now (50%), but simple" << endl;
	cout << "6s /epoch" << endl;
	iEpoch = 0;

	//load and normalize CIFAR10 data
    cout << "Loading CIFAR10 database..." << endl;
    CIFAR10Reader mr;
    if(!mr.read_from_folder(".",mRefImages,mRefLabels, mValImages,mValLabels))
    {
        cout << "CIFAR10 samples not found, please check the CIFAR10 *.bin files are in the executable folder" << endl;
        return -1;
    }
	mValImages/= 256.f;
	mRefImages/= 256.f;
  
	//create simple net:
	net.add(new LayerPoolMax2D(32, 32, 3, 2, 2));
	net.add(new LayerDense(16 * 16 * 3, 256));
	net.add(new LayerActivation("Relu"));
	net.add(new LayerDense(256, 10));
	net.add(new LayerSoftmax());

	//setup train options
	netTrain.set_net(net);
	netTrain.set_epochs(30);
	netTrain.set_batchsize(256);
	netTrain.set_loss("SparseCategoricalCrossEntropy");
	netTrain.set_epoch_callback(epoch_callback); //optional, show the progress
	netTrain.set_train_data(mRefImages, mRefLabels);
	netTrain.set_validation_data(mValImages, mValLabels); //optional, not used for training, helps to keep the final best model

	// train net
	cout << "Training..." << endl << endl;
	start = chrono::steady_clock::now();
	netTrain.train();

	// show train results
	MatrixFloat mClassPredicted;
	net.classify(mRefImages, mClassPredicted);
	ConfusionMatrix cmRef;
	ClassificationResult crRef = cmRef.compute(mRefLabels, mClassPredicted);
	cout << "Ref accuracy: " << crRef.accuracy << " %" << endl;

	MatrixFloat mValClass;
	net.classify(mValImages, mValClass);
	ConfusionMatrix cmValidation;
	ClassificationResult crValidation = cmValidation.compute(mValLabels, mValClass);
	cout << "Validation accuracy: " << crValidation.accuracy << " %" << endl;
	cout << "Test confusion matrix:" << endl << crValidation.mConfMat << endl;

    return 0;
}
