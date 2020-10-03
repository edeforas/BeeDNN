// simple  classification MNIST with aKMeans algorithm
// 91.7 % classification after 20 epochs, 100 centroids are used
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
	auto next = chrono::steady_clock::now();
	auto delta = chrono::duration_cast<std::chrono::milliseconds>(next - start).count();
	start = next;

    iEpoch++;
    cout << "Epoch: " << iEpoch << " duration: " << delta << " ms" << endl;
	cout << " TrainAccuracy: " << kmTrain.get_current_train_accuracy() << " %" ;
	cout << " ValidationAccuracy: " << kmTrain.get_current_validation_accuracy() << " %" << endl;
	cout << " Ref count= " << toString(kmTrain.ref_count().transpose()) << endl;

	cout << endl;
}
//////////////////////////////////////////////////////////////////////////////
int main()
{
	cout << "simple MNIST classification with a KMeans algorithm" << endl;

    iEpoch = 0;

	//load and normalize MNIST data
    cout << "Loading MNIST database..." << endl;
    MNISTReader mr;
    if(!mr.load("."))
    {
        cout << "MNIST samples not found, please check the *.ubyte files are in the executable folder" << endl;
        return -1;
    }
  
	km.set_sizes(784 , 10 * 10); // approximately 10 supports by classes (todo)
	km.set_loss("MeanCubicError"); // gives better accuracy with KMeans

	//setup train options
	kmTrain.set_kmeans(km);
	kmTrain.set_epochs(20);
	kmTrain.set_batchsize(1024);
	kmTrain.set_epoch_callback(epoch_callback); //optional, to show the progress
	kmTrain.set_train_data(mr.train_data(),mr.train_truth());
	kmTrain.set_validation_data(mr.test_data(), mr.test_truth()); //optional

	// train net
	cout << "Training..." << endl << endl;
	start = chrono::steady_clock::now();
	kmTrain.train();

	// show train results
	MatrixFloat mClassPredicted;
	km.predict(mr.train_data(), mClassPredicted);
	ConfusionMatrix cmRef;
	ClassificationResult crRef = cmRef.compute(mr.train_truth(), mClassPredicted);
	cout << "Train accuracy: " << crRef.accuracy << " %" << endl;

	MatrixFloat mClassTest;
	km.predict(mr.test_data(), mClassTest);
	ConfusionMatrix cmVal;
	ClassificationResult crVal = cmVal.compute(mr.test_truth(), mClassTest);
	cout << "Validation accuracy: " << crVal.accuracy << " %" << endl;
	cout << "Validation confusion matrix:" << endl << toString(crVal.mConfMat) << endl;

    return 0;
}
