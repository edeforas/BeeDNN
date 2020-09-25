// simple  classification MNIST with a dense layer, similar as :
// https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/_index.ipynb
// validation accuracy > 98.1%, after 20 epochs (2s by epochs)

#include <iostream>
#include <chrono>
using namespace std;

#include "KMeans.h"
#include "KMeansTrain.h"
#include "MNISTReader.h"
#include "ConfusionMatrix.h"


KMeans km;
KMeansTrain kmTrain;

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
/*	cout << "TrainLoss: " << netTrain.get_current_train_loss() << " TrainAccuracy: " << netTrain.get_current_train_accuracy() << " %" ;
	cout << " ValidationAccuracy: " << netTrain.get_current_validation_accuracy() << " %" << endl;
	*/
	cout << endl;
}
//////////////////////////////////////////////////////////////////////////////
int main()
{
	cout << "simple  classification MNIST with a KMeans algorithm" << endl;
//	cout << "validation accuracy > 98.1%, after 15 epochs (2s by epochs)" << endl;

    iEpoch = 0;

	//load and normalize MNIST data
    cout << "Loading MNIST database..." << endl;
    MNISTReader mr;
    if(!mr.load("."))
    {
        cout << "MNIST samples not found, please check the *.ubyte files are in the executable folder" << endl;
        return -1;
    }
  
	//setup train options
	kmTrain.set_kmeans(km);
	kmTrain.set_epochs(100);
	kmTrain.set_epoch_callback(epoch_callback); //optional, to show the progress
	kmTrain.set_train_data(mr.train_data(),mr.train_truth());
	kmTrain.set_validation_data(mr.test_data(), mr.test_truth()); //optional, not used for training, helps to keep the final best model

	// train net
	cout << "Training..." << endl << endl;
	start = chrono::steady_clock::now();
	kmTrain.train();

	// show train results
	MatrixFloat mClassPredicted;
	km.classify(mr.train_data(), mClassPredicted);
	ConfusionMatrix cmRef;
	ClassificationResult crRef = cmRef.compute(mr.train_truth(), mClassPredicted);
	cout << "Train accuracy: " << crRef.accuracy << " %" << endl;

	MatrixFloat mClassTest;
	km.classify(mr.test_data(), mClassTest);
	ConfusionMatrix cmVal;
	ClassificationResult crVal = cmVal.compute(mr.test_truth(), mClassTest);
	cout << "Validation accuracy: " << crVal.accuracy << " %" << endl;
	cout << "Validation confusion matrix:" << endl << toString(crVal.mConfMat) << endl;
	/*
	//testu function
	if (crVal.accuracy < 98.1f)
	{
		cout << "Test failed! accuracy=" << crVal.accuracy << endl;
		return -1;
	}
	*/
	cout << "Test succeded." << endl;
    return 0;
}
