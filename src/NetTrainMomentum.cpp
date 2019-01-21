#include "NetTrainMomentum.h"
#include "Net.h"
#include "Layer.h"
#include "Matrix.h"
#include "MatrixUtil.h"

#include <cmath>
using namespace std;

/////////////////////////////////////////////////////////////////////////////////////////////////
NetTrainMomentum::NetTrainMomentum()
{ }
/////////////////////////////////////////////////////////////////////////////////////////////////
NetTrainMomentum::~NetTrainMomentum()
{ }
/////////////////////////////////////////////////////////////////////////////////////////////////
void NetTrainMomentum::train(Net& net,const MatrixFloat& mSamples,const MatrixFloat& mTruth,const TrainOption& topt)
{
    size_t nLayers=net.layers().size();

    if(topt.initWeight)
    {
        for(unsigned int i=0;i<nLayers;i++)
            net.layer(i)->initWeights();
    }

    //TrainResult tr;
    unsigned int iBatchSize=topt.batchSize;
    unsigned int iNbSamples=mSamples.rows();

    // init error accumulation and momentum
    vector<MatrixFloat> sumDE, sumDEMomentum;
    for(size_t i=0;i<nLayers;i++)
    {
        sumDE.push_back(net.layer(i)->dE*0); //todo something cleaner
        sumDEMomentum.push_back(net.layer(i)->dE*0);
    }

    for(int iEpoch=0;iEpoch<topt.epochs;iEpoch++)
    {
        //double dMaxError=0., dMeanError=0.; // todo read early abort strategy

        MatrixFloat mShuffle=rand_perm(iNbSamples);

        unsigned int iBatchStart=0;
        while(iBatchStart<iBatchSize)
        {
            unsigned int iBatchEnd=iBatchStart+iBatchSize;
            if(iBatchEnd>iBatchSize)
                iBatchEnd=iBatchSize;

            // init error accumulation
            for(unsigned int i=0;i<nLayers;i++)
                sumDE[i].set_zero();

            for(unsigned int iSample=iBatchStart;iSample<iBatchEnd;iSample++)
            {
                //compute error, sample by sample

                //forward pass with save
                int iIndexSample=(int)mShuffle(iSample);
                const MatrixFloat& mSample=mSamples.row(iIndexSample);
                MatrixFloat mOut, mTemp=mSample;
                for(unsigned int i=0;i<nLayers;i++)
                {
                    net.layer(i)->forward_save(mTemp,mOut);
                    mTemp=mOut; //todo avoid using a temp MatrixFloat
                }

                //compute and backpropagate error, sum dE
                MatrixFloat mError=mOut-mTruth.row(iIndexSample);

                backpropagation(net,mError); //todo move forward?

                //sum error
                for(unsigned int i=0;i<nLayers;i++)
                    sumDE[i]+=net.layer(i)->dE;
            }

            //update weight
            for(unsigned int i=0;i<nLayers;i++)
            {
                Layer& l=*net.layer(i);

                sumDE[i]/=(float)(iBatchEnd-iBatchStart);

                //update weight with momentum: weight -= learning_rate*dE+momentum*oldDE;
                l.get_weight()-=sumDE[i]*topt.learningRate+sumDEMomentum[i]*topt.momentum;

                // update momentum
                sumDEMomentum[i]=sumDE[i];
            }
            iBatchStart=iBatchEnd;
        }

        if(topt.observer)
            topt.observer->stepEpoch(/*tr*/);
    }
}
/////////////////////////////////////////////////////////////////////////////////////////////////
void NetTrainMomentum::backpropagation(Net& net,const MatrixFloat &mError) //todo add input mean for each layer?
{
    MatrixFloat mOne(1,1); mOne.set_constant(1.);

    size_t nLayers=net.layers().size();

    MatrixFloat mDelta;
    for (int i=(int)(nLayers-1);i>=0;i--)
    {
        Layer& l=*net.layer(i);
        MatrixFloat mAD=l.get_weight_activation_derivation();

        if(i==(int)(nLayers-1))
        {
            //last layer
            mDelta=mError.element_product(mAD); //  delta=error.*mAD;
        }
        else
        {
            //hidden layer
            //a=  delta*(net.layer{i+1}.weight') (use of previous delta)
            MatrixFloat a=mDelta*(net.layer(i+1)->get_weight().transpose());

            //a=a(:,1:columns(a)-1); % do not use last weight (use only for bias)
            //b=activation_derivation(layer.func,outweight);
            //delta=a.*b;
            mDelta=(a.without_last_column()).element_product(mAD); //todo optimise
        }

        //dE=(layer.in')*delta;
        l.dE=(l.in.concat(mOne).transpose())*mDelta;
    }
}
/////////////////////////////////////////////////////////////////////////////////////////////
