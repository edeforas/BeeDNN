
#include "Net.h"
#include "Matrix.h"
/////////////////////////////////////////////////////////////////////////////////////////////////
Net::Net()
{}
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
void Net::forward(const Matrix& mIn,Matrix& mOut) const
{
    Matrix mTemp=mIn;
    for(unsigned int i=0;i<_layers.size();i++)
    {
        _layers[i]->forward(mTemp,mOut);
        mTemp=mOut;
    }
}
/////////////////////////////////////////////////////////////////////////////////////////////////
void Net::forward_feed(const Matrix& mIn,Matrix& mOut)
{
    Matrix mTemp=mIn;
    for(unsigned int i=0;i<_layers.size();i++)
    {
        _layers[i]->forward_feed(mTemp,mOut);
        mTemp=mOut;
    }
}
/////////////////////////////////////////////////////////////////////////////////////////////////
TrainResult Net::train(const Matrix& mSamples,const Matrix& mTruth,const TrainOption& topt,bool bInit)
{
    TrainResult tr;

    //for now, no minibatch, no momentum, no early abort, no samples shuffle, just plain batch sgd learning
    if(bInit)
        for(unsigned int i=0;i<_layers.size();i++)
        {
            _layers[i]->init_weight();
        }

    for(int iEpoch=0;iEpoch<topt.epochs;iEpoch++)
    {
        for(unsigned int i=0;i<_layers.size();i++)
            _layers[i]->init_DE();

        //max_error=0

        for(int iSample=0;iSample<mSamples.rows();iSample++)
        {
            //compute one sample error
            Matrix mOut;
            const Matrix& mSample=mSamples.row(iSample);
            forward_feed(mSample,mOut);
            Matrix mError=mOut-mTruth.row(iSample);

            //now backpropagate error, sum dE
            backpropagation(mError,topt.learningRate);//todo

            //update error ??
        }

        //update weight
        for(unsigned int iL=0;iL<_layers.size();iL++)
        {
            Layer& l=*_layers[iL];
            l.get_weight()-=l.dE.scalarMult(topt.learningRate);
            //            net.layer{i}.dE=netaccum.layer{i}.dE+net.momentum*net.layer{i}.dE;
            //            net.layer{i}.weight=net.layer{i}.weight-net.learning_rate*net.layer{i}.dE;

        }
    }

    return tr;
}
/////////////////////////////////////////////////////////////////////////////////////////////////
void Net::backpropagation(const Matrix &mError, double dlearningRate)
{
    Matrix mDelta;
    for (int iL=_layers.size()-1;iL>=0;iL--)
    {
        Layer& l=*_layers[iL];
        Matrix mAD=l.get_weight_activation_derivation();

        if((unsigned int)iL==_layers.size()-1)
        {
            //last layer
            mDelta=mError.elementProduct(mAD); //  delta=error.*mAD;
        }
        else
        {
            //hidden layer
            //a=  delta*(net.layer{i+1}.weight') ;%*  delta; % use of previous delta
            Matrix a=mDelta*(_layers[iL+1]->get_weight().transpose());

            //a=a(:,1:columns(a)-1); % do not use last weight (use only for bias)
            //b=activation_derivation(layer.func,outweight);
            //delta=a.*b;
            mDelta=(a.without_last_column()).elementProduct(mAD);
        }

        //dE=(layer.in')*delta;
        Matrix mOne(1,1); mOne.setConstant(1);
        Matrix mDE=(l.in.concat(mOne).transpose())*mDelta;

        //net.layer{i}.dE=dE;
        l.dE+=mDE;

        //net.layer{i}.weight=net.layer{i}.weight-learning_rate*dE;
        l.get_weight()-=mDE.scalarMult(dlearningRate);
    }
}
/////////////////////////////////////////////////////////////////////////////////////////////
