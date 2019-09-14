// sample  classification MNIST similar as :
// https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/_index.ipynb

//near 98% on ref and test after 10 epochs (3s by epochs)

#include <cstdlib>

#include <iostream>
#include <chrono>
using namespace std;

#include "Net.h"
#include "NetTrain.h"
#include "MNISTReader.h"
#include "ConfusionMatrix.h"

Net net;
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

    cout << endl;
}
//////////////////////////////////////////////////////////////////////////////
int main()
{
    iEpoch = 0;

	//load MNIST data
    cout << "Loading MNIST database..." << endl;
    MNISTReader mr;
    if(!mr.read_from_folder(".",mRefImages,mRefLabels, mTestImages,mTestLabels))
    {
        cout << "MNIST samples not found, please check the *.ubyte files are in the executable folder" << endl;
        return -1;
    }

	//normalize data
	mTestImages/= 256.f;
	mRefImages/= 256.f;
  
	//create simple net:
    net.add_dense_layer(784, 128);
	net.add_activation_layer("Relu");
	net.add_dropout_layer(128,0.1f);
	net.add_dense_layer(128, 10);
	net.add_softmax_layer();

	//train net
	cout << "Training..." << endl <<endl;
	NetTrain netTrain;
	netTrain.set_epochs(10);
	netTrain.set_batchsize(64);

	netTrain.set_loss("CategoricalCrossEntropy");
	netTrain.set_epoch_callback(epoch_callback);
	netTrain.set_train_data(mRefImages, mRefLabels);
	start = chrono::steady_clock::now();
	netTrain.train(net);

	MatrixFloat mClassRef;
	net.predict(mRefImages, mClassRef);
	ConfusionMatrix cmRef;
	ClassificationResult crRef = cmRef.compute(mRefLabels, mClassRef);
	cout << "Ref accuracy: " << crRef.accuracy << " %" << endl;

	MatrixFloat mClassTest;
	net.predict(mTestImages, mClassTest);
	ConfusionMatrix cmTest;
	ClassificationResult crTest = cmTest.compute(mTestLabels, mClassTest);
	cout << "Test accuracy: " << crTest.accuracy << " %" << endl;

	// the end, results are computed and displayed in epoch_callback
	cout << "end of test." << endl;
    return 0;
}
