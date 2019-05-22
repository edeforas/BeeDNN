// this sample shows how to do a simple classification, the usecase is to learn a XOR gate
// the output of this network is a Sigmoid, so the we can use the loss BinaryCrossEntropy and 0 -> class_0 ; 1 -> class_1

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
	net.add_activation_layer("Sigmoid");

    //set the train data
    float dSamples[]={ 0,0 , 0,1 , 1,0 , 1,1 };
    float dTruths[]={ 0 , 1 , 1, 0 };
    const MatrixFloat mSamples=fromRawBuffer(dSamples,4,2);
    const MatrixFloat mTruth=fromRawBuffer(dTruths,4,1);

    //optimize
	NetTrain netFit;
	netFit.set_epochs(2000);
	netFit.set_optimizer("Nadam");
	netFit.set_loss("BinaryCrossEntropy");
	netFit.fit(net,mSamples,mTruth);

    //predict and show results
    MatrixFloat mOut;
	net.forward(mSamples, mOut);
    cout << "0_xor_0=" << mOut(0) << endl << "0_xor_1=" << mOut(1) << endl << "1_xor_0=" << mOut(2) << endl << "1_xor_1=" << mOut(3) << endl;

    //compute and show the total loss
    float fLoss=netFit.compute_loss(net,mSamples,mTruth);
    cout << "BinaryCrossEntropy_Loss=" << fLoss << endl;

    return 0;
}
