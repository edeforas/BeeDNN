#include <iostream>
using namespace std;

#include "Net.h"
#include "NetTrain.h"

int main()
{
    //contruct layer
    Net net;
    net.add_dense_layer(2,3);
	net.add_activation_layer("Relu");
	net.add_dense_layer(3, 1);

    //set train data
    float dSamples[]={ 0,0 , 0,1 , 1,0 , 1,1 };
    float dTruths[]={ 0 , 1 , 1, 0 };
    const MatrixFloat mSamples=fromRawBuffer(dSamples,4,2);
    const MatrixFloat mTruth=fromRawBuffer(dTruths,4,1);

    //learn
    TrainOption tOpt;
	tOpt.epochs = 1000;
	NetTrain netFit;
	netFit.fit(net,mSamples,mTruth,tOpt);

    //predict and show results
    MatrixFloat mPredicted;
    net.forward(mSamples,mPredicted);
    cout << "0xor0=" << mPredicted(0) << " 0xor1=" << mPredicted(1) << " 1xor0=" << mPredicted(2) << " 1xor1=" << mPredicted(3) << endl;

    //compute and show loss
    float fLoss=netFit.compute_loss(net,mSamples,mTruth);
    cout << "Loss=" << fLoss << endl;

    return 0;
}
