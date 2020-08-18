// simple  classification MNIST with a dense layer, similar as :
// https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/_index.ipynb
// validation accuracy > 98.1%, after 15 epochs (2s by epochs)

#include <iostream>
#include <chrono>
using namespace std;

#include "Net.h"
#include "NetTrain.h"
#include "MNISTReader.h"
#include "ConfusionMatrix.h"

#include "LayerActivation.h"
#include "LayerDense.h"
#include "LayerDropout.h"
#include "LayerSoftmax.h"

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
	cout << "simple  classification MNIST with a dense layer" << endl;
	cout << "validation accuracy > 98.1%, after 15 epochs (2s by epochs)" << endl;

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
	net.add(new LayerDense(784, 256));
	net.add(new LayerActivation("Relu"));
	net.add(new LayerDropout(0.2f)); //reduce overfitting
	net.add(new LayerDense(256, 10));
	net.add(new LayerSoftmax());

	//setup train options
	netTrain.set_net(net);
	netTrain.set_epochs(15);
	netTrain.set_batchsize(64);
	netTrain.set_loss("SparseCategoricalCrossEntropy");
	netTrain.set_epoch_callback(epoch_callback); //optional , to show the progress
	netTrain.set_train_data(mr.train_data(),mr.train_truth());
	netTrain.set_validation_data(mr.test_data(), mr.test_truth()); //optional, not used for training, helps to keep the final best model

	// train net
	cout << "Training..." << endl << endl;
	start = chrono::steady_clock::now();
	netTrain.train();

	// show train results
	MatrixFloat mClassPredicted;
	net.classify(mRefImages, mClassPredicted);
	ConfusionMatrix cmRef;
	ClassificationResult crRef = cmRef.compute(mr.train_truth(), mClassPredicted);
	cout << "Train accuracy: " << crRef.accuracy << " %" << endl;

	MatrixFloat mClassTest;
	net.classify(mValImages, mClassTest);
	ConfusionMatrix cmVal;
	ClassificationResult crVal = cmVal.compute(mr.train_truth(), mClassTest);
	cout << "Validation accuracy: " << crVal.accuracy << " %" << endl;
	cout << "Validation confusion matrix:" << endl << toString(crVal.mConfMat) << endl;

	//testu function
	if (crVal.accuracy < 98.1f)
	{
		cout << "Test failed! accuracy=" << crVal.accuracy << endl;
		return -1;
	}

	cout << "Test succeded." << endl;
    return 0;
}
