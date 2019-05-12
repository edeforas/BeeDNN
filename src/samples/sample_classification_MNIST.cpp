// sample  classification MNIST as in:
// https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/_index.ipynb

#include <iostream>
#include <chrono>
using namespace std;

#include "Net.h"
#include "NetTrain.h"
#include "MNISTReader.h"
#include "ConfusionMatrix.h"

Net net;
MatrixFloat mRefImages, mRefLabelsIndex, mTestImages, mTestLabelsIndex;
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
    ClassificationResult crRef = cmRef.compute(mRefLabelsIndex, mClassRef, 10);
    cout << "% accuracy on Ref =" << crRef.accuracy << endl;

    MatrixFloat mClassTest;
    net.classify_all(mTestImages, mClassTest);
    ConfusionMatrix cmTest;
    ClassificationResult crTest = cmTest.compute(mTestLabelsIndex, mClassTest, 10);
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
    if(!mr.read_from_folder(".",mRefImages,mRefLabelsIndex, mTestImages,mTestLabelsIndex))
    {
        cout << "MNIST samples not found, please check the *-ubyte files are in the executable folder" << endl;
        return -1;
    }

    //normalize data
    mTestImages/=255.f;
    mRefImages/=255.f;

	//check perf was 15s/iteration in sample
    
	//create simple net:
    net.add_dense_layer(784,16); //was 512
	net.add_activation_layer("Relu");
	net.add_dropout_layer(16,0.2f);
	net.add_dense_layer(16, 10);
	net.add_activation_layer("Relu"); 
	net.add_softmax_layer();

	//train net
	cout << "training..." << endl;
	TrainOption tOpt;
    tOpt.epochCallBack = epoch_callback;
	tOpt.epochs = 5;

	NetTrain netTrain;
	netTrain.set_loss("CategoricalCrossEntropy");
	netTrain.set_optimizer("Adam");

	start = chrono::steady_clock::now();
    netTrain.train(net,mRefImages,mRefLabelsIndex,tOpt);

	// the end, results are computed and displayed in the callback
	cout << "end of test." << endl;
    return 0;
}
