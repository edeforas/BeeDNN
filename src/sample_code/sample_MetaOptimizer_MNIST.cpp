//This sample launch in parallel multiple runs of the same net optimization 
//This sample also test for many different Relu activations flavors" << endl;
//It shows and save the current best solution on disk
//This is a heavy test, but expect val_accuracy>99.30% after 40min (got max 99.41%)
//To stop by anytime, type CTRL+C

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

#include "NetUtil.h" //for net file saving

//////////////////////////////////////////////////////////////////////////////
void better_solution_callback(NetTrain& train)
{
	cout << "Better solution found: Accuracy= " << train.get_current_validation_accuracy() << endl;

	// save solution to file using a string buffer
	string s;
	NetUtil::write(train,s); //save train parameters
	NetUtil::write(train.net(),s); // save network
	std::ofstream f("solution_accuracy" + to_string(train.get_current_validation_accuracy()) + ".txt");
	f << s;
}
//////////////////////////////////////////////////////////////////////////////
int main()
{
	cout << "This sample launch in parallel multiple runs of the same net optimization" << endl;
	cout << "This sample also test for many different Relu activations flavors" << endl;
	cout << "It shows and save the current best solution on disk" << endl;
	cout << "This is a heavy test, but expect val_accuracy>99.30% after 40min (got max 99.41%)" << endl;
	cout << "To stop by anytime, type CTRL+C" << endl << endl;

	//load MNIST data
	MatrixFloat mRefImages, mRefLabels, mValImages, mValLabels;
	cout << "Loading MNIST database..." << endl;
    MNISTReader mr;
    if(!mr.read_from_folder(".",mRefImages,mRefLabels, mValImages, mValLabels))
    {
        cout << "MNIST samples not found, please check the *-ubyte files are in the executable folder" << endl;
        return -1;
    }

	//normalize pixels data
	mValImages /= 256.f;
	mRefImages /= 256.f;

	//create convolutional net
	Net net;
	net.add(new LayerConvolution2D(28, 28, 1, 3, 3, 8));
	net.add(new LayerChannelBias(26,26,8)); //for now, conv bias is separated
	net.add(new LayerActivation("Relu"));

	net.add(new LayerConvolution2D(26, 26, 8, 3, 3, 8, 2, 2));
	net.add(new LayerChannelBias(12,12,8));
	net.add(new LayerActivation("Relu"));
	net.add(new LayerDropout(0.3f)); //avoid overfitting

	net.add(new LayerConvolution2D(12, 12, 8, 3, 3, 8));
	net.add(new LayerChannelBias(10,10,8));
	net.add(new LayerActivation("Relu"));
	net.add(new LayerDropout(0.3f));

	net.add(new LayerDense(10 * 10 * 8, 128)); //decision layer

	net.add(new LayerActivation("Relu"));
	net.add(new LayerDense(128, 10));
	net.add(new LayerSoftmax());

	//set train settings
	NetTrain netTrain;
	netTrain.set_epochs(50);
	netTrain.set_loss("SparseCategoricalCrossEntropy");
	netTrain.set_train_data(mRefImages, mRefLabels);
	netTrain.set_validation_data(mValImages, mValLabels);
	netTrain.set_net(net);

	//create meta optimizer to run in parallel
	MetaOptimizer optim;
	optim.set_train(netTrain);

	//add Relu variations
	optim.add_variation(2, "RRelu");
	optim.add_variation(2, "PRelu");
	optim.add_variation(2, "LeakyTwiceRelu6");
	optim.add_variation(2, "Relu6");
	optim.add_variation(2, "LeakyRelu");

	optim.add_variation(5, "RRelu");
	optim.add_variation(5, "PRelu");
	optim.add_variation(5, "LeakyTwiceRelu6");
	optim.add_variation(5, "Relu6");
	optim.add_variation(5, "LeakyRelu");

	optim.add_variation(9, "RRelu");
	optim.add_variation(9, "PRelu");
	optim.add_variation(9, "LeakyTwiceRelu6");
	optim.add_variation(9, "Relu6");
	optim.add_variation(9, "LeakyRelu");

	optim.add_variation(12, "RRelu");
	optim.add_variation(12, "PRelu");
	optim.add_variation(12, "LeakyTwiceRelu6");
	optim.add_variation(12, "Relu6");
	optim.add_variation(12, "LeakyRelu");

	optim.set_repeat_all(10); //re-do everything 10 times
	optim.set_better_solution_callback(better_solution_callback); //called on better solution found

	cout << "Training with all CPU cores ..." << endl;
	optim.run(); // will use 100% CPU

	// the end
	cout << "End of test." << endl;
    return 0;
}