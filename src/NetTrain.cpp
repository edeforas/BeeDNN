/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#include "NetTrain.h"

#include "Net.h"
#include "Layer.h"
#include "Matrix.h"
#include "Optimizer.h"

/////////////////////////////////////////////////////////////////////////////////////////////////
NetTrain::NetTrain()
{ }
/////////////////////////////////////////////////////////////////////////////////////////////////
NetTrain::~NetTrain()
{ }
/////////////////////////////////////////////////////////////////////////////////////////////////
float NetTrain::compute_loss(const Net& net, const MatrixFloat &mSamples, const MatrixFloat &mTruth)
{
    if(net.layers().size()==0)
        return -1.;

    int iNbSamples=(int)mSamples.rows();
    MatrixFloat mOut,mError;

    for(int i=0;i<iNbSamples;i++)
    {
        net.forward(mSamples.row(i),mOut);

        if(i==0)
            mError=(mOut-mTruth.row(i)).cwiseAbs2();
        else
            mError+=(mOut-mTruth.row(i)).cwiseAbs2();
    }

    return mError.sum()/iNbSamples;
}
/////////////////////////////////////////////////////////////////////////////////////////////////
TrainResult NetTrain::train(Net& net,const MatrixFloat& mSamples,const MatrixFloat& mTruthLabel,const TrainOption& topt)
{
    if(net.layers().size()==0)
        return TrainResult(); //nothing to do

    bool bOutputIsLabel=net.layer(net.layers().size()-1).out_size()==1;
    int iMax=(int)mTruthLabel.maxCoeff();

    if(!bOutputIsLabel)
    {
        MatrixFloat mTruth(mTruthLabel.rows(),iMax+1);
        mTruth.setZero();
        for(int i=0;i<mTruth.rows();i++)
            mTruth(i,(int)mTruthLabel(i,0))=1;
        return fit(net,mSamples,mTruth,topt);
    }
    else
        return fit(net,mSamples,mTruthLabel,topt);
}
/////////////////////////////////////////////////////////////////////////////////////////////////
TrainResult NetTrain::fit(Net& net,const MatrixFloat& mSamples,const MatrixFloat& mTruth,const TrainOption& topt)
{
    TrainResult tr;
    int iNbSamples=(int)mSamples.rows();
    int nLayers=(int)net.layers().size();

    Net bestNet;

    int iBatchSize=topt.batchSize;
    if(iBatchSize>iNbSamples)
        iBatchSize=iNbSamples;

    float fInvBatchSize=1.f/(float)iBatchSize;

    vector<MatrixFloat> inOut(nLayers+1);
    vector<MatrixFloat> inOutSum(nLayers+1);
    vector<MatrixFloat> deltaSum(nLayers+1);

    vector<Optimizer*> optimizers(nLayers);

    MatrixFloat mLoss;
    double dLoss=0.,dMinLoss=1.e99;

    tr.reset();

    if(nLayers==0)
        return tr;

    for (int i = 0; i < nLayers; i++)
    {
        optimizers[i] = get_optimizer(topt.optimizer);
        optimizers[i]->fLearningRate = topt.learningRate;
        optimizers[i]->fMomentum=topt.momentum;
        optimizers[i]->fDecay=topt.decay;
        optimizers[i]->init(net.layer(i));
    }

    for(int iEpoch=0;iEpoch<topt.epochs;iEpoch++)
    {
        dLoss=0.;

        MatrixFloat mShuffle=randPerm(iNbSamples);
        net.set_train_mode(true);

        int iBatchStart=0;

        while(iBatchStart<iNbSamples)
        {
            int iBatchEnd=iBatchStart+iBatchSize;
            if(iBatchEnd>iNbSamples)
                iBatchEnd=iNbSamples;

            //init all layers inputs sum and error sum
            for(int i=0;i<nLayers+1;i++)
                inOutSum[i].setZero();
            for(int i=0;i<nLayers+1;i++)
                deltaSum[i].setZero();

            for(int iBatch=iBatchStart;iBatch<iBatchEnd;iBatch++)
            {
                int iIndexSample=(int)mShuffle(iBatch,0);
                const MatrixFloat& mSample=mSamples.row(iIndexSample);
                const MatrixFloat& mTarget=mTruth.row(iIndexSample);

                //forward pass with store
                inOut[0]=mSample;
                for(int i=0;i<nLayers;i++)
                    net.layer(i).forward(inOut[i],inOut[i+1]);

                //add all layers inputs
                for(int i=0;i<nLayers+1;i++)
                {
                    if(inOutSum[i].size())
                        inOutSum[i]+=inOut[i];
                    else
                        inOutSum[i]=inOut[i];
                }

                // add all errors
                mLoss=inOut[nLayers]-mTarget;
                dLoss += mLoss.cwiseAbs2().sum(); //update loss
                if(deltaSum[nLayers].size())
                    deltaSum[nLayers]+=mLoss;
                else
                    deltaSum[nLayers]=mLoss;
            }

            //average minibatch
          //  for(int i=0;i<nLayers+1;i++)
          //  {
           //    inOutSum[i]*=fInvBatchSize;
           //    deltaSum[i]*=fInvBatchSize;
          //  }

             deltaSum[nLayers]*=fInvBatchSize;

            //backward pass
            for (int i=(int)(nLayers-1);i>=0;i--)
            {
                Layer& l=net.layer(i);
                l.backpropagation(inOutSum[i], deltaSum[i+1], optimizers[i], deltaSum[i]);
            }

            iBatchStart=iBatchEnd;
        }

        net.set_train_mode(false);
        dLoss/=iNbSamples;

        tr.loss.push_back(dLoss);

        if (topt.epochCallBack)
            topt.epochCallBack();

        //keep the best model if asked
        if(topt.keepBest)
        {
            if(dMinLoss>dLoss)
            {
                dMinLoss=dLoss;
                bestNet=net;
            }
        }
    }

    for (int i = 0; i < nLayers; i++)
        delete optimizers[i];

    if(topt.keepBest)
    {
        net=bestNet;
        tr.finalLoss=dMinLoss;
    }
    else
        tr.finalLoss=dLoss;

    return tr;
}
/////////////////////////////////////////////////////////////////////////////////////////////
