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
    net.add_dense_layer(1,100);
    net.add_activation_layer("Relu");
    net.add_dense_layer(100,1);

    //set train data
    MatrixFloat mTruth(128,1);
    MatrixFloat mSamples(128,1);
    for(int i=0;i<128;i++)
    {
        float x=i/100.f;
        mTruth(i,0)=sin(x);
        mSamples(i,0)=x;
    }

    //train
    cout << "Training..." << endl;
    TrainOption tOpt;
	NetTrain netfit;
	netfit.fit(net,mSamples,mTruth,tOpt);

    //display some sin prediction
	MatrixFloat mPredict;
	net.forward(mSamples, mPredict);
    for(int i=0;i<mSamples.size();i+=8)
        cout << std::setprecision(4) << "x=" << mSamples(i,0) << "\ttruth=" << mTruth(i,0) << "\tpredict=" << mPredict(i,0) <<endl;

    //compute loss
    float fLoss=netfit.compute_loss(net,mSamples,mTruth);
    cout << "Loss=" << fLoss << endl;

    return 0;
}
