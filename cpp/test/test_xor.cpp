#include <iostream>
using namespace std;

#include "Net.h"
#include "ActivationSigmoid.h"
#include "DenseLayer.h"

int main()
{
    Net n;

    DenseLayer l1(2,3,ActivationSigmoid());
    DenseLayer l2(3,1,ActivationSigmoid());

    n.add(&l1);
    n.add(&l2);

    double dSamples[]={ 0 , 0 , 0 , 1 , 1 , 0 , 1 , 1};
    double dTruths[]={ 0 , 1 , 1, 0 };

    const Matrix mSamples(dSamples,4,2);
    const Matrix mTruth(dTruths,4,1);

    TrainOption tOpt;
    tOpt.epochs=5000;
    //tOpt.maxError= 0.05;
    tOpt.learningRate=0.1;

    TrainResult tr=n.train(mSamples,mTruth,tOpt);
    cout << "Loss=" << tr.loss << " MaxError=" << tr.maxError << " MaxEpoch=" << tr.maxEpoch << endl;

    Matrix m00,m01,m10,m11;

    n.forward(mSamples.row(0),m00);
    n.forward(mSamples.row(1),m01);
    n.forward(mSamples.row(2),m10);
    n.forward(mSamples.row(3),m11);
    cout << m00(0)<< " " <<m01(0) << " " << m10(0) << " " << m11(0) << endl;

    return 0;
}
