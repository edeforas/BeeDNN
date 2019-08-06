#include <iostream>
using namespace std;

#include "Net.h"
#include "NetTrain.h"
#include "MNISTReader.h"
#include "MetaOptimizer.h"

//////////////////////////////////////////////////////////////////////////////
int main()
{
	//load MNIST data
	MatrixFloat mRefImages, mRefLabels, mTestImages, mTestLabels; 
	cout << "Loading MNIST database..." << endl;
    MNISTReader mr;
    if(!mr.read_from_folder(".",mRefImages,mRefLabels, mTestImages,mTestLabels))
    {
        cout << "MNIST samples not found, please check the *-ubyte files are in the executable folder" << endl;
        return -1;
    }
  
	//decimate learning data for quicker results in this sample (bad in real case)
	mRefImages = decimate(mRefImages, 10);
	mRefLabels= decimate(mRefLabels, 10);
	mTestImages = decimate(mTestImages, 10);
	mTestLabels = decimate(mTestLabels, 10);

	//normalize pixels data
	mTestImages /= 256.f;
	mRefImages /= 256.f;

	//create simple net
	Net net;
    net.add_dense_layer(784, 32);
	net.add_activation_layer("Relu");
	net.add_dense_layer(32, 10);
	net.add_softmax_layer();

	//train net
	cout << "Training..." << endl;
	NetTrain netTrain;
	netTrain.set_epochs(10);
	netTrain.set_loss("CategoricalCrossEntropy");

	netTrain.set_learning_data(mRefImages, mRefLabels);

	//create meta optimizer
	MetaOptimizer optim;
	optim.set_net(&net);
	optim.set_train(&netTrain);
	optim.run();

	//todo collect results and take the best
	//todo accept variations in meta parameters

	//WIP WIP

	// the end
	cout << "End of test." << endl;
    return 0;
}
