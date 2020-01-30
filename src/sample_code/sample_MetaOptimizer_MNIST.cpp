//this sample launch in parallel multiple runs of the same net optimization 
//and save the current best solution on disk
//this is a heavy test, but expect val_accuracy>99.30% after 30min

#include <iostream>
#include <fstream>
using namespace std;

#include "Net.h"
#include "NetTrain.h"
#include "MNISTReader.h"
#include "MetaOptimizer.h"

#include "LayerActivation.h"
#include "LayerConvolution2D.h"
#include "LayerChannelBias.h"
#include "LayerDense.h"
#include "LayerDropout.h"
#include "LayerSoftmax.h"

#include "NetUtil.h" //for net saving

//////////////////////////////////////////////////////////////////////////////
void better_solution_callback(NetTrain& train)
{
	cout << "Better solution found: Accuracy= " << train.get_current_test_accuracy() << endl;

	// save solution to disk using a string buffer
	string s;
	NetUtil::write(train,s); //save train parameters
	NetUtil::write(train.net(),s); // save network
	std::ofstream f("solution_accuracy" + to_string(train.get_current_test_accuracy()) + ".txt");
	f << s;
}
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
  
	//normalize pixels data
	mTestImages /= 256.f;
	mRefImages /= 256.f;

	//create conv net
	Net net;
	net.add(new LayerConvolution2D(28, 28, 1, 3, 3, 8));
	net.add(new LayerChannelBias(26,26,8));
	net.add(new LayerActivation("Relu"));

	net.add(new LayerConvolution2D(26, 26, 8, 3, 3, 8, 2, 2));
	net.add(new LayerChannelBias(12,12,8));
	net.add(new LayerActivation("Relu"));
	net.add(new LayerDropout(0.3f));

	net.add(new LayerConvolution2D(12, 12, 8, 3, 3, 8));
	net.add(new LayerChannelBias(10,10,8));
	net.add(new LayerActivation("Relu"));
	net.add(new LayerDropout(0.3f));

	net.add(new LayerDense(10 * 10 * 8, 128));

	net.add(new LayerActivation("Relu"));
	net.add(new LayerDense(128, 10));
	net.add(new LayerSoftmax());

	//set train settings
	NetTrain netTrain;
	netTrain.set_epochs(50);
	netTrain.set_loss("SparseCategoricalCrossEntropy");
	netTrain.set_train_data(mRefImages, mRefLabels);
	netTrain.set_test_data(mTestImages, mTestLabels);
	netTrain.set_net(net);

	//create meta optimizer and run in parallel (for now, only weights variations)
	cout << "Training with all cores ..." << endl;
	MetaOptimizer optim; 
	optim.set_train(netTrain);
	optim.set_better_solution_callback(better_solution_callback);
	optim.run(); // will use 100% CPU

	// the end
	cout << "End of test." << endl;
    return 0;
}