#include <iostream>
#include <cmath>
using namespace std;

#include "Net.h"
#include "ActivationTanh.h"
#include "DenseLayer.h"

int main()
{
    Net n;

    ActivationTanh ac;
    DenseLayer l1(1,20,&ac);
    DenseLayer l2(20,20,&ac);
    DenseLayer l3(20,1,&ac);

    n.add(&l1);
    n.add(&l2);
    n.add(&l3);

	Matrix mTruth(64);
	Matrix mSamples(64);
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
