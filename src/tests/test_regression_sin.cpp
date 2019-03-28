#include <iostream>
#include <iomanip>
#include <cmath>
using namespace std;

#include "Net.h"
#include "NetTrain.h"

int main()
{
    //build net
    Net net;
    net.add_dense_layer(1,100,true);
    net.add_activation_layer("Relu");
    net.add_dense_layer(100,1,true);

    //set train data
    MatrixFloat mTruth(128,1);
    MatrixFloat mSamples(128,1);
    for(int i=0;i<128;i++)
    {
        float x=i/100.f;
        mTruth(i,0)=sin(x);
        mSamples(i,0)=x;
    }

    //learn
    cout << "Learning..." << endl;
    TrainOption tOpt;
	NetTrain netfit;
	netfit.fit(net,mSamples,mTruth,tOpt);

    //show some results
    MatrixFloat mOnePredict(1,1), mOneSample(1,1), mOneTruth(1,1);
    for(int i=0;i<mSamples.size();i+=8)
    {
        mOneSample(0,0)=mSamples(i,0);
        mOneTruth(0,0)=mTruth(i,0);
        net.forward(mOneSample,mOnePredict);
        cout << std::setprecision(4) << "x=" << mOneSample(0,0) << "\ttruth=" << mOneTruth(0,0) << "\tpredict=" << mOnePredict(0,0) <<endl;
    }

    //compute loss
    float fLoss=netfit.compute_loss(net,mSamples,mTruth);
    cout << "Loss=" << fLoss << endl;

    return 0;
}
