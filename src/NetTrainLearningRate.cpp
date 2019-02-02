#include "NetTrainLearningRate.h"

#include "Net.h"
#include "Layer.h"
#include "Matrix.h"
#include "MatrixUtil.h"

#include <iostream>
using namespace std;

/////////////////////////////////////////////////////////////////////////////////////////////////
NetTrainLearningRate::NetTrainLearningRate()
{ }
/////////////////////////////////////////////////////////////////////////////////////////////////
NetTrainLearningRate::~NetTrainLearningRate()
{ }
/////////////////////////////////////////////////////////////////////////////////////////////////
void NetTrainLearningRate::train(Net& net,const MatrixFloat& mSamples,const MatrixFloat& mTruth,const TrainOption& topt)
{
    size_t nLayers=net.layers().size();

    if(topt.initWeight)
    {
        for(unsigned int i=0;i<nLayers;i++)
            net.layer(i)->init();
    }

    //TrainResult tr;
    size_t iBatchSize=topt.batchSize;
    size_t iNbSamples=mSamples.rows();

    for(int iEpoch=0;iEpoch<topt.epochs;iEpoch++)
    {
        MatrixFloat mShuffle=rand_perm(iNbSamples);

        size_t iBatchStart=0;
        while(iBatchStart<iBatchSize)
        {
            size_t iBatchEnd=iBatchStart+iBatchSize;
            if(iBatchEnd>iBatchSize) // do not compute with partial minibatch ? todo?
                break;

            //init suminput accumulation
            vector<MatrixFloat> sumInput;
            sumInput.resize(nLayers); //todo keep object memory beetween minbatch, avoid malloc

            MatrixFloat sumDelta; //todo keep object memory beetween minbatch, avoid malloc

            //forward pass, and compute mean layer input and error
            for(size_t iSample=iBatchStart;iSample<iBatchEnd;iSample++)
            {
                //compute total error, sample by sample

                //forward pass with layer input save
                size_t iIndexSample=(size_t)mShuffle(iSample,0);
                const MatrixFloat& mSample=mSamples.row(iIndexSample);
                const MatrixFloat& mTarget=mTruth.row(iIndexSample);
                MatrixFloat mOut, mTemp=mSample;

                //forward pass with store
                for(unsigned int i=0;i<nLayers;i++)
                {
                    net.layer(i)->forward(mTemp,mOut);
                    MatrixFloat& mIn=sumInput[i];
                    if(mIn.size()) //todo correct init
                        mIn+=mTemp;
                    else
                        mIn=mTemp;

                    mTemp=mOut; //todo optim avoid using a temp MatrixFloat
                }

                MatrixFloat mDelta=mOut-mTarget;
                if(sumDelta.size()) // todo correct init
                    sumDelta+=mDelta;
                else
                    sumDelta=mDelta;

            }

            // normalize vs batchsize
            for(unsigned int i=0;i<nLayers;i++)
                sumInput[i]/=(float)iBatchSize;

            //backpropagation of delta and update of weights
            MatrixFloat mNewDelta,mDelta=sumDelta/(float)iBatchSize;

            cout << " " << MatrixUtil::to_string(mDelta); // this is not the error!

            for (int i=(int)(nLayers-1);i>=0;i--)
            {
                Layer* l=net.layer(i);
                l->backpropagation(sumInput[i],mDelta,topt.learningRate,mNewDelta);
				mDelta=mNewDelta; //todo optim avoid resize
            }

            iBatchStart=iBatchEnd;
        }

        if(topt.observer)
            topt.observer->stepEpoch(/*tr*/);
    }
}
/////////////////////////////////////////////////////////////////////////////////////////////
