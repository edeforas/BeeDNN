#include <iostream>
using namespace std;

#include "Net.h"
#include "LayerDenseAndBias.h"
#include "LayerDenseNoBias.h"
#include "LayerActivation.h"
#include "MatrixUtil.h"

#include "NetTrainLearningRate.h"

int main()
{
    //contruct layer
    Net net;
    net.add(new LayerDenseAndBias(2,3));
    net.add(new LayerActivation("Sigmoid"));
    net.add(new LayerDenseAndBias(3,1));

    //train data
    float dSamples[]={ 0,0 , 0,1 , 1,0 , 1,1 };
    float dTruths[]={ 0 , 1 , 1, 0 };
    const MatrixFloat mSamples=from_raw_buffer(dSamples,4,2);
    const MatrixFloat mTruth=from_raw_buffer(dTruths,4,1);

    //cout << MatrixUtil::to_string(mTruth) << endl;

    TrainOption tOpt;
    tOpt.learningRate=0.1f;
    tOpt.batchSize=1;
    tOpt.epochs=20000;

    NetTrainLearningRate netTrain;
    netTrain.train(net,mSamples,mTruth,tOpt);
    //cout << "Loss=" << tr.loss << " MaxError=" << tr.maxError << " ComputedEpochs=" << tr.computedEpochs << endl;

    MatrixFloat m00,m01,m10,m11;

    net.forward(mSamples.row(0),m00);
    net.forward(mSamples.row(1),m01);
    net.forward(mSamples.row(2),m10);
    net.forward(mSamples.row(3),m11);

    cout << m00(0) << " " <<m01(0) << " " << m10(0) << " " << m11(0) << endl;

    return 0;
}
