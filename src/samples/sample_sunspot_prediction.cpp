// simple sunspot prediction using a dense layer and windowed data:
// time serie from https://www.kaggle.com/robervalt/sunspots (simplified to one column)

#include <iostream>
#include <chrono>
using namespace std;

#include "Net.h"
#include "NetTrain.h"
#include "CsvFileReader.h"

#include "ConfusionMatrix.h"
#include "LayerActivation.h"
#include "LayerDense.h"
#include "LayerDropout.h"
#include "LayerSoftmax.h"
#include "TimeSeriesUtil.h"

Net net;
NetTrain netTrain;
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
	cout << "TrainLoss: " << netTrain.get_current_train_loss() << endl ;
}
//////////////////////////////////////////////////////////////////////////////
int main()
{
	cout << "Simple sunspot prediction using a dense layer and windowed data" << endl;
	CsvFileReader mr;
	mr.load("datasets_2418_917074_Sunspots_simplified.csv"); //only need train data

	if(mr.train_data().size()==0)
	{
		cout << "datasets_2418_917074_Sunspots_simplified.csv not found, please check it is in the executable folder" << endl;
		return -1;
	}

    iEpoch = 0;

	// convert time series to windowed datas (for simple dense net), not test data here
	int iWindowSize = 20;
	MatrixFloat mDataTrain, mTruthTrain;
	TimeSeriesUtil::generate_windowed_data(mr.train_data(), iWindowSize, mDataTrain);
	mTruthTrain = rowView(mr.train_data(), iWindowSize, mr.train_data().rows()); //the 10 samples in data gives 1 predicted sample in truth, this is why we start at 10

	//create simple net, iWindowSize input, one output, 10 hidden neurons
	net.add(new LayerDense(iWindowSize, 64));
	net.add(new LayerActivation("Relu"));
	net.add(new LayerDropout(0.3f)); //reduce overfitting
	net.add(new LayerDense(64, 64));
	net.add(new LayerActivation("Relu"));
	net.add(new LayerDropout(0.2f)); //reduce overfitting
	net.add(new LayerDense(64, 1));

	//setup train options
	netTrain.set_net(net);
	netTrain.set_epochs(200);
	netTrain.set_batchsize(64);
	netTrain.set_loss("MeanSquaredError");
	netTrain.set_epoch_callback(epoch_callback); //optional , to show the progress
	netTrain.set_train_data(mDataTrain, mTruthTrain);

	// train net
	cout << "Training..." << endl << endl;
	start = chrono::steady_clock::now();
	netTrain.train();

	//now save truth and predicted in a csv file

	//TODO

	cout << "Test succeded." << endl;
	cout << "Open the .csv file to see how the net predicted the sun spot number." << endl;
	cout << "1st column is truth and 2nd column is predicted" << endl;
	return 0;
}
