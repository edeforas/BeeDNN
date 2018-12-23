#include <iostream>
#include <iomanip>
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
    tOpt.epochs=1000;
    tOpt.earlyAbortMaxError=0.05;
    tOpt.learningRate=0.1f;
    tOpt.batchSize=1;
    tOpt.momentum=0.05f;

    cout << "Learning..." << endl;
    int nbEpochs=n.train(mSamples,mTruth,tOpt);
    cout << "nb epochs=" << nbEpochs << endl;

    //show results
    MatrixFloat mOnePredict(1), mOneSample(1), mOneTruth(1);
    for(unsigned int i=0;i<mSamples.size();i+=4) //show 16 samples
    {
        mOneSample(0)=mSamples(i);
        mOneTruth(0)=mTruth(i);
        n.forward(mOneSample,mOnePredict);
        cout << std::setprecision(4) << "x=" << mOneSample(0) << "\ttruth=" <<mOneTruth(0) << "\tpredict=" << mOnePredict(0) <<endl;
    }
    return 0;
}
