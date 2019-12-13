//this sample launch in parallel multiple runs of same net optimization 
//and save the current best solution on disk

#include <iostream>
#include <fstream>
using namespace std;

#include "Net.h"
#include "NetTrain.h"
#include "MNISTReader.h"
#include "MetaOptimizer.h"

#include "NetUtil.h" //for net saving

//////////////////////////////////////////////////////////////////////////////
void better_solution_callback(NetTrain& train)
{
	cout << "Better solution found: Accuracy= " << train.get_current_test_accuracy() << endl;

	// save solution to disk using a string buffer
	string s;
	NetUtil::write(train,s); //save train
	NetUtil::write(train.net(),s); // save net
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

	//create simple net
	Net net;
    net.add_dense_layer(784, 256);
	net.add_activation_layer("Relu");
	net.add_dropout_layer(256,0.2f);
	net.add_dense_layer(256, 10);
	net.add_softmax_layer();

	//set train settings
	NetTrain netTrain;
	netTrain.set_epochs(30);
	netTrain.set_loss("SparseCategoricalCrossEntropy");
	netTrain.set_train_data(mRefImages, mRefLabels);
	netTrain.set_test_data(mTestImages, mTestLabels);
	netTrain.set_net(net);

	//create meta optimizer and run in // (for now, only weights variations)
	cout << "Training with all cores ..." << endl;
	MetaOptimizer optim;
	optim.set_train(netTrain);
	optim.set_better_solution_callback(better_solution_callback);
	optim.run(); // will use 100% CPU

	// the end
	cout << "End of test." << endl;
    return 0;
}