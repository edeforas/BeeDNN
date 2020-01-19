// all convolutional MNIST classification with a conv2d
// accuracy TBD > % after 10 epochs, TBDs/epochs
// conv2d speed not optimized yet, this test takes time!

#include <iostream>
#include <chrono>
using namespace std;

#include "Net.h"
#include "NetTrain.h"
#include "MNISTReader.h"
#include "ConfusionMatrix.h"

NetTrain netTrain;
MatrixFloat mRefImages, mRefLabels, mTestImages, mTestLabels;
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
	cout << " TestAccuracy: " << netTrain.get_current_test_accuracy() << " %" << endl;

	cout << endl;
}
//////////////////////////////////////////////////////////////////////////////
int main()
{
    iEpoch = 0;

	cout << "all convolutional MNIST classification with a conv2d" << endl;
	cout << "accuracy TBD > % after 10 epochs, TBDs/epochs" << endl;
	cout << "conv2d speed not optimized yet, this test takes time! " << endl;

	//load and normalize MNIST data
    cout << "Loading MNIST database..." << endl;
    MNISTReader mr;
    if(!mr.read_from_folder(".",mRefImages,mRefLabels, mTestImages,mTestLabels))
    {
        cout << "MNIST samples not found, please check the *.ubyte files are in the executable folder" << endl;
        return -1;
    }
	mTestImages/= 256.f;
	mRefImages/= 256.f;
  
	//create a all convolutional net:
	Net net;
	net.add_convolution2D_layer(28, 28, 1, 3, 3, 8);
	net.add_bias_layer();
	net.add_activation_layer("Relu");

	net.add_convolution2D_layer(26, 26, 8, 3, 3, 8, 2, 2);
	net.add_bias_layer();
	net.add_activation_layer("Relu");
	
	net.add_convolution2D_layer(12, 12, 8, 3, 3, 8);
	net.add_bias_layer();
	net.add_activation_layer("Relu");
	
	net.add_dense_layer(10 * 10 * 8, 128);
	net.add_activation_layer("Relu");
	net.add_dense_layer(128, 10);
	net.add_softmax_layer();

	//setup train options
	netTrain.set_net(net);
	netTrain.set_epochs(20);
	netTrain.set_batchsize(32);
	netTrain.set_loss("SparseCategoricalCrossEntropy");
	netTrain.set_epoch_callback(epoch_callback); //optional, show progress
	netTrain.set_train_data(mRefImages, mRefLabels);
	netTrain.set_test_data(mTestImages, mTestLabels); //optional, not used for training, helps to keep the final best model

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

	MatrixFloat mClassTest;
	net.classify(mTestImages, mClassTest);
	ConfusionMatrix cmTest;
	ClassificationResult crTest = cmTest.compute(mTestLabels, mClassTest);
	cout << "Test accuracy: " << crTest.accuracy << " %" << endl;
	cout << "Test confusion matrix:" << endl << crTest.mConfMat << endl;
	
	//testu function
	if (crTest.accuracy < 95.f)
	{
		cout << "Test failed! accuracy=" << crTest.accuracy << endl;
		return -1;
	}

	cout << "Test succeded." << endl;
    return 0;
}
