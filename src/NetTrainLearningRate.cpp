#include "NetTrainLearningRate.h"

#include "Net.h"
#include "Layer.h"
#include "Matrix.h"
#include "MatrixUtil.h"

/////////////////////////////////////////////////////////////////////////////////////////////////
NetTrainLearningRate::NetTrainLearningRate(): NetTrain()
{ }
/////////////////////////////////////////////////////////////////////////////////////////////////
NetTrainLearningRate::~NetTrainLearningRate()
{ }
/////////////////////////////////////////////////////////////////////////////////////////////////
void NetTrainLearningRate::train(Net& net,const MatrixFloat& mSamples,const MatrixFloat& mTruth,const TrainOption& topt)
{
    int nLayers=(int)net.layers().size();

    if(nLayers==0)
        return;

    //TrainResult tr;
    int iBatchSize=topt.batchSize;
    int iNbSamples=mSamples.rows();

    for(int iEpoch=0;iEpoch<topt.epochs;iEpoch++)
    {
        MatrixFloat mShuffle=rand_perm(iNbSamples);

        int iBatchStart=0;
        while(iBatchStart<iBatchSize)
        {
            int iBatchEnd=iBatchStart+iBatchSize;
            if(iBatchEnd>iBatchSize)
                iBatchEnd=iBatchSize; //last batch can be partial

            //init suminput accumulation
            vector<MatrixFloat> sumInput;
            sumInput.resize(nLayers); //todo keep object memory beetween minbatch, avoid malloc

            MatrixFloat sumDelta; //todo keep object memory beetween minbatch, avoid malloc

            //forward pass, and compute mean layer input and error
            for(int iSample=iBatchStart;iSample<iBatchEnd;iSample++)
            {
                //compute total error, sample by sample

                //forward pass with layer input save
                int iIndexSample=(int)mShuffle(iSample,0);
                const MatrixFloat& mSample=mSamples.row(iIndexSample);
                const MatrixFloat& mTarget=mTruth.row(iIndexSample);
                MatrixFloat mOut, mTemp=mSample;

                //forward pass with store
                for(int i=0;i<nLayers;i++)
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
            for(int i=0;i<nLayers;i++)
                sumInput[i]/=(float)iBatchSize;

            //backpropagation of delta and update of weights
            MatrixFloat mNewDelta,mDelta=sumDelta/(float)iBatchSize;

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
