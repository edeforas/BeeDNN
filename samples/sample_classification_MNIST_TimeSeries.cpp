// simple MNIST classification, all image seen as a time series rows by rows
// validation accuracy > 98.1%, after 20 epochs (1s by epochs)

#include <iostream>, row by rows
#include <chrono>
using namespace std;

#include "Net.h"
#include "NetTrain.h"
#include "MNISTReader.h"
#include "ConfusionMatrix.h"

#include "LayerActivation.h"
#include "LayerDot.h"
#include "LayerDense.h"
#include "LayerTimeDistributedBias.h"
#include "LayerDropout.h"
#include "LayerSoftmax.h"

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
	cout << "TrainLoss: " << netTrain.get_current_train_loss() << " TrainAccuracy: " << netTrain.get_current_train_accuracy() << " %" ;
	cout << " ValidationAccuracy: " << netTrain.get_current_validation_accuracy() << " %" << endl;

	cout << endl;
}
//////////////////////////////////////////////////////////////////////////////
int main()
{
	cout << "simple MNIST classification, all image seen as a time series rows by rows" << endl;
	cout << "validation accuracy > 98%, after 15 epochs (1s by epochs)" << endl;

    iEpoch = 0;

	//load and normalize MNIST data
    cout << "Loading MNIST database..." << endl;
    MNISTReader mr;
    if(!mr.load("."))
    {
        cout << "MNIST samples not found, please check the *.ubyte files are in the executable folder" << endl;
        return -1;
    }
  
	//create simple net:
	net.add(new LayerTimeDistributedBias(28));
	net.add(new LayerDot(28*28, 128));
	net.add(new LayerActivation("Relu"));
	net.add(new LayerDropout(0.2f)); //reduce overfitting
	net.add(new LayerDense(128, 10));
	net.add(new LayerSoftmax());

	//setup train options
	netTrain.set_net(net);
	netTrain.set_epochs(15);
	netTrain.set_batchsize(64);
	netTrain.set_loss("SparseCategoricalCrossEntropy");
	netTrain.set_epoch_callback(epoch_callback); //optional, to show the progress
	netTrain.set_train_data(mr.train_data(),mr.train_truth());
	netTrain.set_validation_data(mr.validation_data(), mr.validation_truth()); //optional, not used for training, helps to keep the final best model

	// train net
	cout << "Training..." << endl << endl;
	start = chrono::steady_clock::now();
	netTrain.fit();

	// show train results
	MatrixFloat mClassPredicted;
	net.predict_classes(mr.train_data(), mClassPredicted);
	ConfusionMatrix cmRef;
	ClassificationResult crRef = cmRef.compute(mr.train_truth(), mClassPredicted);
	cout << "Train accuracy: " << crRef.accuracy << " %" << endl;

	MatrixFloat mClassVal;
	net.predict_classes(mr.validation_data(), mClassVal);
	ConfusionMatrix cmVal;
	ClassificationResult crVal = cmVal.compute(mr.validation_truth(), mClassVal);
	cout << "Validation accuracy: " << crVal.accuracy << " %" << endl;
	cout << "Validation confusion matrix:" << endl << toString(crVal.mConfMat) << endl;

	//testu function
	if (crVal.accuracy < 98.f)
	{
		cout << "Test failed! accuracy=" << crVal.accuracy << endl;
		return -1;
	}

	cout << "Test succeded." << endl;
    return 0;
}
