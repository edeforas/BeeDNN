#include <iostream>
#include <iomanip>
#include <cmath>
using namespace std;

#include "Net.h"
#include "NetTrainMomentum.h"

#include "Activation.h"
#include "ActivationLayer.h"

int main()
{
    //build net
    Net net;
    net.add(new ActivationLayer(1,20,"Tanh"));
    net.add(new ActivationLayer(20,20,"Tanh"));
    net.add(new ActivationLayer(20,1,"Tanh"));

    //train data
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
    tOpt.learningRate=0.1f;
    tOpt.batchSize=1;
    tOpt.momentum=0.05f;

    cout << "Learning..." << endl;
    NetTrainMomentum train;
    train.train(net,mSamples,mTruth,tOpt);

    //show results
    MatrixFloat mOnePredict(1), mOneSample(1), mOneTruth(1);
    for(unsigned int i=0;i<mSamples.size();i+=4) //show 16 samples
    {
        mOneSample(0)=mSamples(i);
        mOneTruth(0)=mTruth(i);
        net.forward(mOneSample,mOnePredict);
        cout << std::setprecision(4) << "x=" << mOneSample(0) << "\ttruth=" <<mOneTruth(0) << "\tpredict=" << mOnePredict(0) <<endl;
    }
    return 0;
}
