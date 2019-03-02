#include <iostream>
using namespace std;

#include "Net.h"
#include "NetTrainSGD.h"
#include "MNISTReader.h"
#include "ConfusionMatrix.h"

Net net;
MatrixFloat mRefImages, mRefLabelsIndex, mTestImages, mTestLabelsIndex;
int iEpoch;

//////////////////////////////////////////////////////////////////////////////
void epoch_callback()
{
    iEpoch++;
    cout << " epoch:" << iEpoch << endl;

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
        cout << "MNIST samples not found, please check the *-ubyte files are in executable folder" << endl;
        return -1;
    }
/*
    //reduce data size (test)
    mTestImages=decimate(mTestImages,10);
    mRefImages=decimate(mRefImages,10);
    mRefLabelsIndex=decimate(mRefLabelsIndex,10);
    mTestLabelsIndex=decimate(mTestLabelsIndex,10);
*/
    //normalize data
    mTestImages/=256.f;
    mRefImages/=256.f;

    //simple net: 97% classif 76% test after a long time
    net.add_dense_layer("DenseAndBias",784,64);
    net.add_dropout_layer(64,0.2f);
    net.add_activation_layer("Relu");
    net.add_dense_layer("DenseAndBias",64,10);
    net.add_dropout_layer(10,0.2f);
    net.add_activation_layer("Sigmoid");

    TrainOption tOpt;
    tOpt.epochs=100;
    tOpt.epochCallBack = epoch_callback;

    cout << "training..." << endl;

    NetTrainSGD netTrain;
    netTrain.train(net,mRefImages,mRefLabelsIndex,tOpt);

    cout << "end of test." << endl;
    return 0;
}
