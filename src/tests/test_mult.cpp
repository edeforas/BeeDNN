#include <iostream>
using namespace std;

#include "Net.h"
#include "LayerDenseWithBias.h"
#include "LayerActivation.h"

#include "NetTrainLearningRate.h"

int main()
{
    //1D problem

    //contruct layer
    Net net;
    net.add(new LayerDenseWithBias(1,1));
    net.add(new LayerActivation("Tanh"));

    //train data
    float dSamples[]={  -2, 3 ,-0.3 ,-1 ,1 ,20 };

    MatrixFloat mSamples(6,1);
    MatrixFloat mTruth(6,1);

    for(int i=0;i<6;i++)
    {
        mSamples(i,0)=dSamples[i];
        mTruth(i,0)=tanh(dSamples[i]*2.f+1.f);
    }

    TrainOption tOpt;
    tOpt.learningRate=0.001f;
    tOpt.batchSize=1;
    tOpt.epochs=10000;

    NetTrainLearningRate netTrain;
    netTrain.train(net,mSamples,mTruth,tOpt);

    MatrixFloat u;
    for(int i=0;i<6;i++)
    {
        net.forward(mSamples.row(i),u);
        cout << "in=" << mSamples(i,0) << " truth=" << mTruth(i,0) << " predicted=" << u(0)  << endl;
    }

    return 0;
}
