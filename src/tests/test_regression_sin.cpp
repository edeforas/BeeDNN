#include <iostream>
#include <cmath>
using namespace std;

#include "Net.h"
#include "Activation.h"
#include "ActivationLayer.h"

int main()
{
    Net n;

    ActivationLayer l1(1,20,"Tanh");
    ActivationLayer l2(20,20,"Tanh");
    ActivationLayer l3(20,1,"Tanh");

    n.add(&l1);
    n.add(&l2);
    n.add(&l3);

    MatrixFloat mTruth(64);
    MatrixFloat mSamples(64);
	for( int i=0;i<64;i++)
	{
        float x=i/10.f;
		mTruth(i)=sin(x);
        mSamples(i)=x;
	}

    TrainOption tOpt;
    tOpt.epochs=10000;
    tOpt.earlyAbortMaxError=0.05;
    tOpt.learningRate=0.1f;
    tOpt.batchSize=1;
    tOpt.momentum=0.05f;

    TrainResult tr=n.train(mSamples,mTruth,tOpt);
    cout << "Loss=" << tr.loss << " MaxError=" << tr.maxError << " ComputedEpochs=" << tr.computedEpochs << endl;

    return 0;
}
