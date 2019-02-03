#include <iostream>
using namespace std;

#include "Net.h"
#include "LayerDenseAndBias.h"
#include "LayerActivation.h"

#include "NetTrainLearningRate.h"

int main()
{
    //contruct layer
    Net net;
    net.add(new LayerDenseAndBias(1,1));

    //train data
    float dSamples[]={  -2, 3 ,-1 ,4 ,1 ,2 };
    float dTruths[]={ -4+1, 6+1, -2+1, 8+1, 2+1, 4+1 }; //2x+1
    const MatrixFloat mSamples=from_raw_buffer(dSamples,6,1);
    const MatrixFloat mTruth=from_raw_buffer(dTruths,6,1);

    TrainOption tOpt;
    tOpt.learningRate=0.1f;
    tOpt.batchSize=1;
    tOpt.epochs=1000;

    NetTrainLearningRate netTrain;
    netTrain.train(net,mSamples,mTruth,tOpt);

    MatrixFloat u;
    for(int i=0;i<6;i++)
    {
        net.forward(mSamples.row(i),u);
        cout << u(0)<< " " ;
    }

    return 0;
}
