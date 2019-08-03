
#include <iostream>
#include <chrono>
using namespace std;

#include "Net.h"
#include "NetTrain.h"
#include "MNISTReader.h"
#include "ConfusionMatrix.h"
#include "MetaOptimizer.h"

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
	// WIP WIP WIP


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
	mTestImages/= 256.f;
	mRefImages/= 256.f;
  
	//decimate learning data for quicker results in this sample: bad
	mRefImages = decimate(mRefImages, 10);
	mRefLabels= decimate(mRefLabels, 10);

	//create simple net:
    net.add_dense_layer(784, 256);
	net.add_activation_layer("LeakyRelu");
	net.add_dense_layer(256, 10);
	net.add_softmax_layer();

	//train net
	cout << "training..." << endl;
	NetTrain netTrain;
	netTrain.set_epochs(106);
	netTrain.set_loss("CategoricalCrossEntropy");
	netTrain.set_epoch_callback(epoch_callback);
	start = chrono::steady_clock::now();
	netTrain.set_learning_data(mRefImages, mRefLabels);

	MetaOptimizer optim;
	optim.set_net(&net);
	optim.set_train(&netTrain);
	optim.run();

	//todo collect results and take the best
	//todo accept variations

	//WIP WIP

	// the end, results are computed and displayed in epoch_callback
	cout << "end of test." << endl;
    return 0;
}
