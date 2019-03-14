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
double NetTrain::compute_loss(const Net& net, const MatrixFloat &mSamples, const MatrixFloat &mTruth)
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

    return mError.sum()/(double)iNbSamples;
}
/////////////////////////////////////////////////////////////////////////////////////////////
vector<double> NetTrain::loss()
{
    return _vdLoss;
}
/////////////////////////////////////////////////////////////////////////////////////////////////
void NetTrain::train(Net& net,const MatrixFloat& mSamples,const MatrixFloat& mTruthLabel,const TrainOption& topt)
{
    //create a kronecker matrix, one line by sample
    bool bOutputIsLabel=net.layer(net.layers().size()-1)->out_size()==1;
    int iMax=(int)mTruthLabel.maxCoeff();

    if(!bOutputIsLabel)
    {
        MatrixFloat mTruth(mTruthLabel.rows(),iMax+1);
        mTruth.setZero();
        for(int i=0;i<mTruth.rows();i++)
            mTruth(i,(int)mTruthLabel(i,0))=1;
        fit(net,mSamples,mTruth,topt);
    }
    else
        fit(net,mSamples,mTruthLabel,topt);
}
/////////////////////////////////////////////////////////////////////////////////////////////////
void NetTrain::fit(Net& net,const MatrixFloat& mSamples,const MatrixFloat& mTruth,const TrainOption& topt)
{
    int iNbSamples=(int)mSamples.rows();
    int nLayers=(int)net.layers().size();

    int iBatchSize=topt.batchSize;
    if(iBatchSize>iNbSamples)
        iBatchSize=iNbSamples;

    float fInvBatchSize=1.f/(float)iBatchSize;

    vector<MatrixFloat> inOut(nLayers+1);
    vector<MatrixFloat> inOutSum(nLayers+1);
    vector<MatrixFloat> delta(nLayers+1);
    vector<MatrixFloat> deltaSum(nLayers+1);

    vector<Optimizer*> optimizers(nLayers);

    double dLoss=0.;

    _vdLoss.clear();

    if(nLayers==0)
        return;

    for (int i = 0; i < nLayers; i++)
    {
        optimizers[i] = get_optimizer(topt.optimizer);
        optimizers[i]->fLearningRate = topt.learningRate;
        optimizers[i]->fMomentum=topt.momentum;
        optimizers[i]->fDecay=topt.decay;
        optimizers[i]->init(*(net.layer(i)));
    }

    for(int iEpoch=0;iEpoch<topt.epochs;iEpoch++)
    {
        MatrixFloat mShuffle=randPerm(iNbSamples);
        net.set_train_mode(true);

        int iBatchStart=0;

        while(iBatchStart<iNbSamples)
        {
            int iBatchEnd=iBatchStart+iBatchSize;
            if(iBatchEnd>iNbSamples)
                iBatchEnd=iNbSamples;

            //init all layers inputs and error
            for(int i=0;i<nLayers;i++)
                inOutSum[i].setZero();
            for(int i=0;i<nLayers+1;i++)
                deltaSum[i].setZero();

            for(int iBatch=iBatchStart;iBatch<iBatchEnd;iBatch++)
            {
                int iIndexSample=(int)mShuffle(iBatch,0);
                const MatrixFloat& mSample=mSamples.row(iIndexSample);
                const MatrixFloat& mTarget=mTruth.row(iIndexSample);

                //forward pass with store and add
                inOut[0]=mSample;
                for(int i=0;i<nLayers;i++)
                    net.layer(i)->forward(inOut[i],inOut[i+1]);

                //add all layers inputs
                for(int i=0;i<nLayers;i++)
                {
                    if(inOutSum[i].size())
                        inOutSum[i]+=inOut[i];
                    else
                        inOutSum[i]=inOut[i];
                }

                // add all errors
                delta[nLayers]=inOut[nLayers]-mTarget;
                if(deltaSum[nLayers].size())
                    deltaSum[nLayers]+=delta[nLayers];
                else
                    deltaSum[nLayers]=delta[nLayers];
            }

            //backward pass
            deltaSum[nLayers]*=fInvBatchSize;
            for (int i=(int)(nLayers-1);i>=0;i--)
            {
                Layer* l=net.layer(i);
                l->backpropagation(inOutSum[i]*fInvBatchSize,deltaSum[i+1], optimizers[i],deltaSum[i]);
            }

            iBatchStart=iBatchEnd;
        }

        net.set_train_mode(false);

        if (topt.epochCallBack)
            topt.epochCallBack();

        if(topt.testEveryEpochs!=-1)
            if( (iEpoch% topt.testEveryEpochs) == 0)
                dLoss=compute_loss(net,mSamples,mTruth);
        _vdLoss.push_back(dLoss);
    }

    for (int i = 0; i < nLayers; i++)
        delete optimizers[i];
}
/////////////////////////////////////////////////////////////////////////////////////////////
