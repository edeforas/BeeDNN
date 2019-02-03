#include <iostream>
#include <iomanip>
#include <cmath>
using namespace std;

#include "Net.h"
#include "LayerDenseAndBias.h"
#include "LayerDenseNoBias.h"
#include "LayerActivation.h"

#include "NetTrainLearningRate.h"

int main()
{
    //build net
    Net net;
    net.add(new LayerDenseAndBias(1,10));
    net.add(new LayerActivation(10,"Tanh"));
    
    net.add(new LayerDenseAndBias(10,1));
  //  net.add(new LayerActivation(1,"Tanh"));
	
  //  net.add(new LayerDenseWithoutBias(7,1));
  //  net.add(new LayerActivation(1,"Tanh"));

    //train data
    MatrixFloat mTruth(64,1);
    MatrixFloat mSamples(64,1);
    for(size_t i=0;i<64;i++)
    {
        float x=i/10.f;
        mTruth(i,0)=sin(x);
        mSamples(i,0)=x;
    }

    TrainOption tOpt;
    tOpt.epochs=1000;
    tOpt.learningRate=0.01f;
    tOpt.batchSize=64;
  //  tOpt.momentum=0.05f;

    cout << "Learning..." << endl;
    NetTrainLearningRate train;
    train.train(net,mSamples,mTruth,tOpt);

    //show results
    MatrixFloat mOnePredict(1,1), mOneSample(1,1), mOneTruth(1,1);
    for(unsigned int i=0;i<mSamples.size();i+=4) //show 16 samples
    {
        mOneSample(0,0)=mSamples(i,0);
        mOneTruth(0,0)=mTruth(i,0);
        net.forward(mOneSample,mOnePredict);
        cout << std::setprecision(4) << "x=" << mOneSample(0,0) << "\ttruth=" << mOneTruth(0,0) << "\tpredict=" << mOnePredict(0,0) <<endl;
    }
    return 0;
}
