//This sample launch in parallel multiple runs of the same net optimization 
//This sample can also test for many different activations and optimizers
//It shows and save the current best model on disk
//To stop by anytime, type CTRL+C

#include <iostream>
#include <iomanip>
#include <fstream>
using namespace std;

#include "Net.h"
#include "NetTrain.h"
#include "MNISTReader.h"
#include "MetaOptimizer.h"

#include "LayerActivation.h"
#include "LayerDense.h"
#include "LayerDropout.h"
#include "LayerSoftmax.h"

#include "NetUtil.h" //for net file saving

//////////////////////////////////////////////////////////////////////////////
void better_model_callback(NetTrain& train)
{
	cout << "Better model found: Accuracy= " << train.get_current_validation_accuracy() << endl;

	// save solution
	ostringstream sFile;
	sFile << "model_accuracy" << fixed << setprecision(2) << train.get_current_validation_accuracy() << ".json";
	NetUtil::save(sFile.str(),train.net(),train); //save train parameters and net
}
//////////////////////////////////////////////////////////////////////////////
int main()
{
	cout << "This sample launch in parallel multiple runs of the same net optimization" << endl;
	cout << "This sample can also test for many different activations and optimizers" << endl;
	cout << "It shows and save the current best model on disk" << endl;
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

	//simple net for a quicker test
	Net net;
	net.add(new LayerDense(784, 128));
	net.add(new LayerActivation("Relu"));
	net.add(new LayerDropout(0.2f)); //reduce overfitting
	net.add(new LayerDense(128, 10));
	net.add(new LayerSoftmax());

	//set train settings
	NetTrain netTrain;
	netTrain.set_epochs(50);
	netTrain.set_loss("SparseCategoricalCrossEntropy");
	netTrain.set_train_data(mr.train_data(), mr.train_truth());
	netTrain.set_validation_data(mr.validation_data(),mr.validation_truth());

	//create meta optimizer
	MetaOptimizer optim;
	optim.set_train(netTrain);

	//add activations variations
	optim.add_layer_variation(1, "Relu");
	optim.add_layer_variation(1, "RRelu");
	optim.add_layer_variation(1, "PRelu");
	optim.add_layer_variation(1, "LeakyRelu");
	optim.add_layer_variation(1, "Swish");
	optim.add_layer_variation(1, "LogSigmoid");
	optim.add_layer_variation(1, "Tanh");
	optim.add_layer_variation(1, "HardTanh");
	optim.add_layer_variation(1, "Sigmoid");
	optim.add_layer_variation(1, "Mish");
	
	// add optimizer variations
	optim.add_optimizer_variation("AdamW", 0.01f);
	optim.add_optimizer_variation("Adam", 0.01f);
	optim.add_optimizer_variation("Nadam", 0.01f);

	optim.set_repeat_all(10); //re-do everything 10 times
	optim.set_better_model_callback(better_model_callback); //called on better solution found

	cout << "Training with all CPU cores ..." << endl;
	optim.fit(net); // will use 100% CPU

	// the end
	cout << "End of test." << endl;
    return 0;
}