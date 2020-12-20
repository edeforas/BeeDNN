// Sample MNIST classification with a poolmax2D
// 97% accuracy after 25 epochs, 1s/epochs
// image is "undersampled" at first layer with a poolmax2d

#include <iostream>
#include <chrono>
using namespace std;

#include "Net.h"
#include "NetTrain.h"
#include "MNISTReader.h"
#include "ConfusionMatrix.h"

#include "LayerActivation.h"
#include "LayerDense.h"
#include "LayerPoolMax2D.h"
#include "LayerSoftmax.h"

Net net;
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
	cout << "Sample MNIST classification with a poolmax2D" << endl;
	cout << "97% accuracy after 20 epochs, 1s/epochs" << endl;
	cout << "image is undersampled at first layer with a poolmax2d" << endl;

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
	net.add(new LayerPoolMax2D(28,28,1, 2, 2)); //input rows, input cols,input channels, factor rows, factor cols
	net.add(new LayerDense(784/4, 64)); // new size is 4x smaller
	net.add(new LayerActivation("Relu"));
	net.add(new LayerDense(64, 10));
	net.add(new LayerSoftmax());

	//setup train options
	netTrain.set_net(net);
	netTrain.set_epochs(25);
	netTrain.set_batchsize(64);
	netTrain.set_loss("SparseCategoricalCrossEntropy");
	netTrain.set_epoch_callback(epoch_callback); //optional, show progress
	netTrain.set_train_data(mr.train_data(), mr.train_truth());
	netTrain.set_validation_data(mr.validation_data(),mr.validation_truth()); //optional, not used for training, helps to keep the final best model

	// train net
	cout << "Training..." << endl << endl;
	start = chrono::steady_clock::now();
	netTrain.fit();

	// show train results
	MatrixFloat mClassPredicted;
	net.predict_classes(mr.train_data(), mClassPredicted);
	ConfusionMatrix cmRef;
	ClassificationResult crRef = cmRef.compute(mr.train_truth(), mClassPredicted);
	cout << "Train accuracy: " << crRef.accuracy << " %" << endl;

	MatrixFloat mClassVal;
	net.predict_classes(mr.validation_data(), mClassVal);
	ConfusionMatrix cmVal;
	ClassificationResult crVal = cmVal.compute(mr.validation_truth(), mClassVal);
	cout << "Val accuracy: " << crVal.accuracy << " %" << endl;
	cout << "Val confusion matrix:" << endl << toString(crVal.mConfMat) << endl;
	
	//testu function
	if (crVal.accuracy < 96.f)
	{
		cout << "Test failed! accuracy=" << crVal.accuracy << endl;
		return -1;
	}

	cout << "Test succeded." << endl;
    return 0;
}
