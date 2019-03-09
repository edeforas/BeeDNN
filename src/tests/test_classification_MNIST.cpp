#include <iostream>
#include <chrono>
using namespace std;

#include "Net.h"
#include "NetTrainSGD.h"
#include "MNISTReader.h"
#include "ConfusionMatrix.h"

Net net;
MatrixFloat mRefImages, mRefLabelsIndex, mTestImages, mTestLabelsIndex;
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
    cout << " epoch:" << iEpoch << " duration:" << delta << " ms" << endl;

    MatrixFloat mClassRef;
    net.classify_all(mRefImages, mClassRef);
    ConfusionMatrix cmRef;
    ClassificationResult crRef = cmRef.compute(mRefLabelsIndex, mClassRef, 10);
    cout << "% accuracy on Ref =" << crRef.goodclassificationPercent << endl;

    MatrixFloat mClassTest;
    net.classify_all(mTestImages, mClassTest);
    ConfusionMatrix cmTest;
    ClassificationResult crTest = cmTest.compute(mTestLabelsIndex, mClassTest, 10);
    cout << "% accuracy on Test=" << crTest.goodclassificationPercent << endl;

    cout << endl;
}
//////////////////////////////////////////////////////////////////////////////
int main()
{
    iEpoch = 0;

    cout << "loading MNIST database..." << endl;
    MNISTReader mr;
    if(!mr.read_from_folder(".",mRefImages,mRefLabelsIndex, mTestImages,mTestLabelsIndex))
    {
        cout << "MNIST samples not found, please check the *-ubyte files are in the executable folder" << endl;
        return -1;
    }

    //normalize data
    mTestImages/=256.f;
    mRefImages/=256.f;

    //simple net: 97% classif 76% test after a long time
    net.add_dense_layer(784,64,true);
    net.add_dropout_layer(64,0.2f);
    net.add_activation_layer("Relu");
    net.add_dense_layer(64,10,true);
    net.add_dropout_layer(10,0.2f);
    net.add_activation_layer("Sigmoid");

    TrainOption tOpt;
    tOpt.epochCallBack = epoch_callback;

    cout << "training..." << endl;

    NetTrainSGD netTrain;
	start = chrono::steady_clock::now();
    netTrain.train(net,mRefImages,mRefLabelsIndex,tOpt);

    cout << "end of test." << endl;
    return 0;
}
