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

    _vdLoss.clear();
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
         //   vector<MatrixFloat> sumInput(nLayers);
            vector<MatrixFloat> inOut(nLayers+1);
            vector<MatrixFloat> sumInOut(nLayers+1);
          //  sumInput.resize(nLayers); //todo keep object memory beetween minbatch, avoid malloc

            MatrixFloat sumDelta; //todo keep object memory beetween minibatch, avoid malloc

            //forward pass, and compute mean layer input and error
            bool bInitBatch=true;
            for(int iSample=iBatchStart;iSample<iBatchEnd;iSample++)
            {
                //compute total error, sample by sample

                //forward pass with layer input save
                int iIndexSample=(int)mShuffle(iSample,0);
                const MatrixFloat& mSample=mSamples.row(iIndexSample);
                inOut[0]=mSample;

                if(bInitBatch)
                    sumInOut[0]=mSample;
                else
                    sumInOut[0]+=mSample;

                const MatrixFloat& mTarget=mTruth.row(iIndexSample);
                MatrixFloat mOut;//, mTemp=mSample;

                //forward pass with store and add
                for(int i=0;i<nLayers;i++)
                {
                    net.layer(i)->forward(inOut[i],inOut[i+1]);

                    if(bInitBatch)
                        sumInOut[i+1]=inOut[i+1];
                    else
                        sumInOut[i+1]+=inOut[i+1];
                }

                MatrixFloat mDelta=inOut[nLayers]-mTarget;
                if(bInitBatch)
                    sumDelta=mDelta;
                else
                    sumDelta+=mDelta;

                bInitBatch=false;
            }

            // normalize vs batchsize
            for(int i=0;i<nLayers;i++)
                sumInOut[i]/=(float)iBatchSize;

            //backpropagation of delta and update of weights
            MatrixFloat mNewDelta,mDelta=sumDelta/(float)iBatchSize;

            for (int i=(int)(nLayers-1);i>=0;i--)
            {
                Layer* l=net.layer(i);
                l->backpropagation(sumInOut[i],mDelta,topt.learningRate,mNewDelta);
				mDelta=mNewDelta; //todo optim avoid resize
            }

            iBatchStart=iBatchEnd;
        }

        if(topt.observer)
            topt.observer->stepEpoch(/*tr*/);

        _vdLoss.push_back(compute_loss(net,mSamples,mTruth));
    }
}
/////////////////////////////////////////////////////////////////////////////////////////////
