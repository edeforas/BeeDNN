#include <iostream>
using namespace std;

#include "Net.h"
#include "Activation.h"
#include "ActivationLayer.h"

int main()
{
    Net n;

    Layer* l1=new ActivationLayer(2,3,"Sigmoid");
    Layer* l2=new ActivationLayer(3,1,"Sigmoid");

    n.add(l1);
    n.add(l2);

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
    n.train(mSamples,mTruth,tOpt);
    //cout << "Loss=" << tr.loss << " MaxError=" << tr.maxError << " ComputedEpochs=" << tr.computedEpochs << endl;

    MatrixFloat m00,m01,m10,m11;

    n.forward(mSamples.row(0),m00);
    n.forward(mSamples.row(1),m01);
    n.forward(mSamples.row(2),m10);
    n.forward(mSamples.row(3),m11);
    cout << m00(0)<< " " <<m01(0) << " " << m10(0) << " " << m11(0) << endl;

    return 0;
}
