#include "Net.h"
#include "Layer.h"
#include "Matrix.h"
#include "MatrixUtil.h"

#include <iostream>
#include <chrono>
#include <cmath>
using namespace std;

/////////////////////////////////////////////////////////////////////////////////////////////////
Net::Net()
{ }
/////////////////////////////////////////////////////////////////////////////////////////////////
Net::~Net()
{
    //no ownership of layers: nothing to delete
}
/////////////////////////////////////////////////////////////////////////////////////////////////
void Net::add(Layer* l)
{
    _layers.push_back(l);
}
/////////////////////////////////////////////////////////////////////////////////////////////////
void Net::forward(const MatrixFloat& mIn,MatrixFloat& mOut) const
{
    MatrixFloat mTemp=mIn;
    for(unsigned int i=0;i<_layers.size();i++)
    {
        _layers[i]->forward(mTemp,mOut);
        mTemp=mOut; //todo avoid resize
    }
}
/////////////////////////////////////////////////////////////////////////////////////////////////
void Net::classify(const MatrixFloat& mIn,MatrixFloat& mClass) const
{
    // todo merge with forward
    mClass.resize(mIn.rows(),1); //todo  put int NetClassification problem

    MatrixFloat mOut;
    for(unsigned int i=0;i<mIn.rows();i++)
    {
        forward(mIn.row(i),mOut);
        mClass(i)=argmax(mOut)(0); //weird
    }
}
/////////////////////////////////////////////////////////////////////////////////////////////////
TrainResult Net::train(const MatrixFloat& mSamples,const MatrixFloat& mTruth,const TrainOption& topt)
{
    TrainResult tr;

    for(unsigned int i=0;i<_layers.size();i++)
        _layers[i]->init();

    unsigned int iBatchSize=topt.batchSize;
    unsigned int iNbSamples=mSamples.rows();
    unsigned int iNbSamplesSubSampled=iNbSamples/topt.subSamplingRatio;
    if(iBatchSize>iNbSamplesSubSampled)
        iBatchSize=iNbSamplesSubSampled;

    // init error accumulation and momentum
    vector<MatrixFloat> sumDE, sumDEMomentum;
    for(unsigned int i=0;i<_layers.size();i++)
    {
        sumDE.push_back(_layers[i]->dE*0); //todo something cleaner
        sumDEMomentum.push_back(_layers[i]->dE*0);
    }

    for(int iEpoch=0;iEpoch<topt.epochs;iEpoch++)
    {
        //compute epoch duration
        auto beginDuration = std::chrono::steady_clock::now();

        double dMaxError=0., dMeanError=0.;

        MatrixFloat mShuffle=rand_perm(iNbSamples);

        unsigned int iBatchStart=0;
        while(iBatchStart<iNbSamplesSubSampled)
        {
            // init error accumulation
            for(unsigned int i=0;i<_layers.size();i++)
                sumDE[i].set_zero();

            unsigned int iBatchEnd=iBatchStart+iBatchSize;
            if(iBatchEnd>iNbSamplesSubSampled)
                iBatchEnd=iNbSamplesSubSampled;

            for(unsigned int iSample=iBatchStart;iSample<iBatchEnd;iSample++)
            {
                //compute one sample error

                //forward pass with save
                int iIndexSample=(int)mShuffle(iSample);
                const MatrixFloat& mSample=mSamples.row(iIndexSample);
                MatrixFloat mOut, mTemp=mSample;
                for(unsigned int i=0;i<_layers.size();i++)
                {
                    _layers[i]->forward_save(mTemp,mOut);
                    mTemp=mOut; //todo avoid using a temp MatrixFloat
                }

                //compute and backpropagate error, sum dE
                MatrixFloat mError=mOut-mTruth.row(iIndexSample);

                // check early abort max error
                for(unsigned int i=0;i<mError.size();i++)
                {
                    if(fabs(mError(i))>dMaxError)
                        dMaxError=fabs(mError(i));

                    dMeanError+=fabs(mError(i));
                }

                backpropagation(mError);

                //sum error
                for(unsigned int i=0;i<_layers.size();i++)
                    sumDE[i]+=_layers[i]->dE;
            }

            //update weight
            for(unsigned int iL=0;iL<_layers.size();iL++)
            {
                Layer& l=*_layers[iL];

                sumDE[iL]/=(float)(iBatchEnd-iBatchStart);

                //update weight with momentum: weight -= learning_rate*dE+momentum*oldDE;
                l.get_weight()-=sumDE[iL]*topt.learningRate+sumDEMomentum[iL]*topt.momentum;

                // update momentum
                sumDEMomentum[iL]=sumDE[iL];
            }

            iBatchStart=iBatchEnd;
        }

        auto endDuration = chrono::steady_clock::now();

        //early abort test on error
        tr.computedEpochs=iEpoch+1;
        tr.epochDuration=chrono::duration_cast<chrono::microseconds> (endDuration-beginDuration).count()/1.e6;
        tr.maxError=dMaxError;
        tr.loss=dMeanError/(iNbSamplesSubSampled*mTruth.size()); //same as mean error?

        if(topt.observer)
            topt.observer->stepEpoch(tr);

        if( dMaxError<topt.earlyAbortMaxError)
            break;

        if (tr.loss<topt.earlyAbortMeanError)
            break;
    }

    return tr;
}
/////////////////////////////////////////////////////////////////////////////////////////////////
void Net::backpropagation(const MatrixFloat &mError)
{
    MatrixFloat mDelta;
    for (int iL=(int)(_layers.size()-1);iL>=0;iL--)
    {
        Layer& l=*_layers[iL];
        MatrixFloat mAD=l.get_weight_activation_derivation();

        if(iL==(int)(_layers.size()-1))
        {
            //last layer
            mDelta=mError.element_product(mAD); //  delta=error.*mAD;
        }
        else
        {
            //hidden layer
            //a=  delta*(net.layer{i+1}.weight') (use of previous delta)
            MatrixFloat a=mDelta*(_layers[iL+1]->get_weight().transpose());

            //a=a(:,1:columns(a)-1); % do not use last weight (use only for bias)
            //b=activation_derivation(layer.func,outweight);
            //delta=a.*b;
            mDelta=(a.without_last_column()).element_product(mAD); //todo optimise
        }

        //dE=(layer.in')*delta;
        MatrixFloat mOne(1,1); mOne.set_constant(1.);
        l.dE=(l.in.concat(mOne).transpose())*mDelta;
    }
}
/////////////////////////////////////////////////////////////////////////////////////////////
