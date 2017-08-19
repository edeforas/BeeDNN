#include <iostream>
#include <cmath>
using namespace std;

#include "Net.h"
#include "Activation.h"
#include "DenseLayer.h"

int main()
{
    Net n;

    ActivationManager am;
    DenseLayer l1(1,20,am.get_activation("Tanh"));
    DenseLayer l2(20,20,am.get_activation("Tanh"));
    DenseLayer l3(20,1,am.get_activation("Tanh"));

    n.add(&l1);
    n.add(&l2);
    n.add(&l3);

    MatrixFloat mTruth(64);
    MatrixFloat mSamples(64);
	for( int i=0;i<64;i++)
	{
		double x=(double)i/10.;
		mTruth(i)=sin(x);
		mSamples(i)=x;
	}

    TrainOption tOpt;
    tOpt.epochs=10000;
    tOpt.earlyAbortMaxError=0.05;
    tOpt.learningRate=0.1;
    tOpt.batchSize=1;
    tOpt.momentum=0.05;

    TrainResult tr=n.train(mSamples,mTruth,tOpt);
    cout << "Loss=" << tr.loss << " MaxError=" << tr.maxError << " ComputedEpochs=" << tr.computedEpochs << endl;

    return 0;
}
