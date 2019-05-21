// sample  classification MNIST similar as :
// https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/_index.ipynb

//expect 97% on ref and test in less than 100s training time

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
    cout << " epoch:" << iEpoch << " duration:" << delta << " ms" << endl;

    MatrixFloat mClassRef;
    net.classify_all(mRefImages, mClassRef);
    ConfusionMatrix cmRef;
    ClassificationResult crRef = cmRef.compute(mRefLabels, mClassRef);
    cout << "% accuracy on Ref =" << crRef.accuracy << endl;
	
    MatrixFloat mClassTest;
    net.classify_all(mTestImages, mClassTest);
    ConfusionMatrix cmTest;
    ClassificationResult crTest = cmTest.compute(mTestLabels, mClassTest);
    cout << "% accuracy on Test=" << crTest.accuracy << endl;

    cout << endl;
}
//////////////////////////////////////////////////////////////////////////////
int main()
{
    iEpoch = 0;

	//load MNIST data
    cout << "loading MNIST database..." << endl;
    MNISTReader mr;
    if(!mr.read_from_folder(".",mRefImages,mRefLabels, mTestImages,mTestLabels))
    {
        cout << "MNIST samples not found, please check the *-ubyte files are in the executable folder" << endl;
        return -1;
    }

	//normalize data
	mTestImages/= 255.f;
	mRefImages/= 255.f;
  
	//create simple net:
    net.add_dense_layer(784,64);
	net.add_activation_layer("Relu");
	net.add_dense_layer(64, 10);
	net.add_activation_layer("Sigmoid");

	//train net
	cout << "training..." << endl;
	TrainOption tOpt;
    tOpt.epochCallBack = epoch_callback;

	NetTrain netTrain;
	netTrain.set_epochs(10);
	start = chrono::steady_clock::now();
	netTrain.train(net, mRefImages, mRefLabels,tOpt);

	// the end, results are computed and displayed in the callback
	cout << "end of test." << endl;
    return 0;
}
