#include <iostream>
using namespace std;

#include "Net.h"
#include "NetTrainMomentum.h"

#include "Activation.h"
#include "ActivationLayer.h"

int main()
{
    //build layer
    Net net;
    net.add(new ActivationLayer(2,3,"Sigmoid"));
    net.add(new ActivationLayer(3,1,"Sigmoid"));

    //train data
    float dSamples[]={ 0 , 0 , 0 , 1 , 1 , 0 , 1 , 1};
    float dTruths[]={ 0 , 1 , 1, 0 };
    const MatrixFloat mSamples(dSamples,4,2);
    const MatrixFloat mTruth(dTruths,4,1);

    TrainOption tOpt;
    tOpt.earlyAbortMaxError=0.05;
    tOpt.learningRate=1.f;
    tOpt.batchSize=1;
    tOpt.momentum=0.9f;

    //  TrainResult tr=
    NetTrainMomentum netTrain;

    netTrain.train(net,mSamples,mTruth,tOpt);
    //cout << "Loss=" << tr.loss << " MaxError=" << tr.maxError << " ComputedEpochs=" << tr.computedEpochs << endl;

    MatrixFloat m00,m01,m10,m11;

    net.forward(mSamples.row(0),m00);
    net.forward(mSamples.row(1),m01);
    net.forward(mSamples.row(2),m10);
    net.forward(mSamples.row(3),m11);
    cout << m00(0)<< " " <<m01(0) << " " << m10(0) << " " << m11(0) << endl;

    return 0;
}
