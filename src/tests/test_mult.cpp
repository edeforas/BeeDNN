#include <iostream>
#include <cmath>
using namespace std;

#include "Net.h"
#include "LayerDenseAndBias.h"
#include "LayerDenseNoBias.h"
#include "LayerActivation.h"

#include "NetTrainLearningRate.h"

int main()
{
    //1D problem

    //contruct layer
    Net net;
    net.add(new LayerDenseAndBias(1,1));
  //  net.add(new LayerActivation("Tanh"));

    //train data
    float dSamples[]={  -2.f, 0.1f ,-0.3f ,0.f,-1.f ,1.f ,20.f };
    int nbSamples=7;

    MatrixFloat mSamples(nbSamples,1);
    MatrixFloat mTruth(nbSamples,1);

    for(int i=0;i<nbSamples;i++)
    {
        mSamples(i,0)=dSamples[i];
        mTruth(i,0)=(dSamples[i]*2.f+3.f/*+1.f*/);
    }

    TrainOption tOpt;
    tOpt.learningRate=0.01f;
    tOpt.batchSize=7;
    tOpt.epochs=10000;

    NetTrainLearningRate netTrain;
    netTrain.train(net,mSamples,mTruth,tOpt);

    MatrixFloat u;
    for(int i=0;i<nbSamples;i++)
    {
        net.forward(mSamples.row(i),u);
        cout << "in=" << mSamples(i,0) << " truth=" << mTruth(i,0) << " predicted=" << u(0)  << endl;
    }

    return 0;
}
