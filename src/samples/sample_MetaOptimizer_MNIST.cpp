//This sample launch in parallel multiple runs of the same net optimization 
//This sample can also test for many different activations flavors or optimizers
//It shows and save the current best solution on disk
//This is a heavy test, but expect val_accuracy>99.20% after 40min
//To stop by anytime, type CTRL+C

#include <iostream>
#include <iostream>
#include <iomanip>
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

	ostringstream sFile;
	sFile << "solution_accuracy" << fixed << setprecision(2) << train.get_current_validation_accuracy() << ".txt";
	std::ofstream f(sFile.str());
	f << s;
}
//////////////////////////////////////////////////////////////////////////////
int main()
{
	cout << "This sample launch in parallel multiple runs of the same net optimization" << endl;
	cout << "This sample can also test for many different Relu activations flavors" << endl;
	cout << "It shows and save the current best solution on disk" << endl;
	cout << "This is a heavy test, but expect val_accuracy>99.20% after 40min" << endl;
	cout << "To stop by anytime, type CTRL+C" << endl;

	//load MNIST data
	MatrixFloat mRefImages, mRefLabels, mValImages, mValLabels;
	cout << "Loading MNIST database..." << endl;
    MNISTReader mr;
    if(!mr.load("."))
    {
        cout << "MNIST samples not found, please check the *-ubyte files are in the executable folder" << endl;
        return -1;
    }

	//create convolutional net
	Net net;
	net.add(new LayerConvolution2D(28, 28, 1, 3, 3, 8));
	net.add(new LayerChannelBias(26,26,8)); //for now, conv bias is split
	net.add(new LayerActivation("Relu"));

	net.add(new LayerConvolution2D(26, 26, 8, 3, 3, 8, 2, 2));
	net.add(new LayerChannelBias(12,12,8));
	net.add(new LayerActivation("Relu"));

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
	netTrain.set_train_data(mr.train_data(), mr.train_truth());
	netTrain.set_validation_data(mr.validation_data(),mr.validation_truth());
	netTrain.set_net(net);

	//create meta optimizer to run in parallel
	MetaOptimizer optim;
	optim.set_train(netTrain);

	//add Relu variations ; uncomment to test
	/*
	//test for other Relu flavors in any layers
	optim.add_layer_variation(2, "RRelu");
	optim.add_layer_variation(2, "PRelu");
	optim.add_layer_variation(2, "LeakyTwiceRelu6");
	optim.add_layer_variation(2, "Relu6");
	optim.add_layer_variation(2, "LeakyRelu");

	optim.add_layer_variation(5, "RRelu");
	optim.add_layer_variation(5, "PRelu");
	optim.add_layer_variation(5, "LeakyTwiceRelu6");
	optim.add_layer_variation(5, "Relu6");
	optim.add_layer_variation(5, "LeakyRelu");

	optim.add_layer_variation(9, "RRelu");
	optim.add_layer_variation(9, "PRelu");
	optim.add_layer_variation(9, "LeakyTwiceRelu6");
	optim.add_layer_variation(9, "Relu6");
	optim.add_layer_variation(9, "LeakyRelu");

	optim.add_layer_variation(12, "RRelu");
	optim.add_layer_variation(12, "PRelu");
	optim.add_layer_variation(12, "LeakyTwiceRelu6");
	optim.add_layer_variation(12, "Relu6");
	optim.add_layer_variation(12, "LeakyRelu");
	
	// add optimizer variations ; uncomment to test
	optim.add_optimizer_variation("SGD", 0.05f);
	optim.add_optimizer_variation("Momentum", 0.02f);
	optim.add_optimizer_variation("AdamW", 0.01f);
	*/

	optim.set_repeat_all(10); //re-do everything 10 times
	optim.set_better_solution_callback(better_solution_callback); //called on better solution found

	cout << "Training with all CPU cores ..." << endl;
	optim.fit(); // will use 100% CPU

	// the end
	cout << "End of test." << endl;
    return 0;
}