#include "NetTrainSGD.h"

#include "Net.h"
#include "Layer.h"
#include "Matrix.h"
#include "MatrixUtil.h"

/////////////////////////////////////////////////////////////////////////////////////////////////
NetTrainSGD::NetTrainSGD(): NetTrain()
{ }
/////////////////////////////////////////////////////////////////////////////////////////////////
NetTrainSGD::~NetTrainSGD()
{ }
/////////////////////////////////////////////////////////////////////////////////////////////////
void NetTrainSGD::fit(Net& net,const MatrixFloat& mSamples,const MatrixFloat& mTruth,const TrainOption& topt)
{
    int iNbSamples=mSamples.rows();
    int nLayers=(int)net.layers().size();
    vector<MatrixFloat> inOut(nLayers+1);
    vector<MatrixFloat> delta(nLayers+1);

    _vdLoss.clear();

    if(nLayers==0)
        return;

    for(int iEpoch=0;iEpoch<topt.epochs;iEpoch++)
    {
        MatrixFloat mShuffle=rand_perm(iNbSamples);

        for(int iSample=0;iSample<iNbSamples;iSample++)
        {
            int iIndexSample=(int)mShuffle(iSample,0);
            const MatrixFloat& mSample=mSamples.row(iIndexSample);
            const MatrixFloat& mTarget=mTruth.row(iIndexSample);

            //forward pass with store and add
            inOut[0]=mSample;
            for(int i=0;i<nLayers;i++)
                net.layer(i)->forward(inOut[i],inOut[i+1]);

            //backward pass
            delta[nLayers]=inOut[nLayers]-mTarget;
            for (int i=(int)(nLayers-1);i>=0;i--)
            {
                Layer* l=net.layer(i);
                l->backpropagation(inOut[i],delta[i+1],topt.learningRate,delta[i]);
            }
        }

        if(topt.observer)
            topt.observer->stepEpoch(/*tr*/);

        _vdLoss.push_back(compute_loss(net,mSamples,mTruth));
    }
}
/////////////////////////////////////////////////////////////////////////////////////////////
