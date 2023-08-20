// simple MNIST classification with a KMeans algorithm
// 92 % classification after 30 epochs, 100 centroids are used, 2s/epoch
#include <iostream>
#include <chrono>
using namespace std;

#include "KMeans.h"
#include "KMeansTrain.h"
#include "MNISTReader.h"
#include "Metrics.h"
using namespace bee;
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

	cout << endl;
}
//////////////////////////////////////////////////////////////////////////////
int main()
{
	cout << "simple MNIST classification with a KMeans algorithm" << endl;
	cout << "validation accuracy > 92.3%, after 10 epochs (5s by epochs)" << endl;

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
	km.set_loss("L3"); // gives better accuracy with KMeans

	//setup train options
	kmTrain.set_kmeans(km);
	kmTrain.set_epochs(10);
	kmTrain.set_batchsize(10000);
	kmTrain.set_epoch_callback(epoch_callback); //optional, to show the progress
	kmTrain.set_train_data(mr.train_data(),mr.train_truth());
	kmTrain.set_validation_data(mr.validation_data(), mr.validation_truth()); //optional

	// train net
	cout << "Training..." << endl << endl;
	start = chrono::steady_clock::now();
	kmTrain.fit();

	// show train results
	MatrixFloat mClassPredicted;
	km.predict_classes(mr.train_data(), mClassPredicted);
	Metrics metricsTrain;
	metricsTrain.compute(mr.train_truth(), mClassPredicted);
	cout << "Train accuracy: " << metricsTrain.accuracy() << " %" << endl;

	MatrixFloat mClassTest;
	km.predict_classes(mr.validation_data(), mClassTest);
	Metrics metricsVal;
	metricsVal.compute(mr.validation_truth(), mClassTest);
	cout << "Validation accuracy: " << metricsVal.accuracy() << " %" << endl;
	cout << "Validation confusion matrix:" << endl << toString(metricsVal.confusion_matrix()) << endl;

    return 0;
}
