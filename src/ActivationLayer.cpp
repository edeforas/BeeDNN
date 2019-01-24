#include "ActivationLayer.h"

#include <cassert>
#include <cstdlib> // for rand
#include <cmath> // for sqrt

#include "Activation.h"

///////////////////////////////////////////////////////////////////////////////
ActivationLayer::ActivationLayer(int iInSize,int iOutSize,string sActivation):
    Layer(iInSize,iOutSize)
{
    _pActiv=get_activation(sActivation);
    assert(_pActiv);
    _weight.resize(_iInSize+1,_iOutSize); //+1 for bias
}
///////////////////////////////////////////////////////////////////////////////
ActivationLayer::~ActivationLayer()
{
    delete _pActiv;
}
///////////////////////////////////////////////////////////////////////////////
void ActivationLayer::initWeights()
{
    float a =sqrtf(6.f/(_iInSize+_iOutSize));

    if(_pActiv->name()=="Sigmoid")
        a*=4.f;

    for(unsigned int i=0;i<_weight.size();i++)
    {
        _weight(i)=((float)rand()/(float)RAND_MAX-0.5f)*2.f*a;
    }

    dE.resize(_iInSize+1,_iOutSize);
    dE.setZero();
}
///////////////////////////////////////////////////////////////////////////////
void ActivationLayer::forward(const MatrixFloat& mMatIn,MatrixFloat& mMatOut) const
{
    // compute out=[in 1]*weight; todo use MAC
    mMatOut=mMatIn*(without_last_row(_weight));
    mMatOut+=_weight.row(_iInSize);

    // apply activation
    for(unsigned int i=0;i<mMatOut.size();i++)
    {
        mMatOut(i)=_pActiv->apply(mMatOut(i)); //todo keep MatrixFloat in layer, do not resize
    }
}
///////////////////////////////////////////////////////////////////////////////
void ActivationLayer::forward_save(const MatrixFloat& mMatIn,MatrixFloat& mMatOut)
{
    //todo remove this, use forward() instead
    in=mMatIn;

    // compute out=[in 1]*weight; todo use MAC
    mMatOut=mMatIn*without_last_row(_weight);
    mMatOut+=_weight.row(_iInSize);

    outWeight=mMatOut;

    // apply activation
    for(unsigned int i=0;i<mMatOut.size();i++)
    {
        mMatOut(i)=_pActiv->apply(mMatOut(i)); //todo keep MatrixFloat in layer, do not resize
    }
    out=mMatOut;
}
///////////////////////////////////////////////////////////////////////////////
MatrixFloat ActivationLayer::get_weight_activation_derivation() const
{
    // apply activation derivation on outWeight
    MatrixFloat mOut=outWeight;
    for(unsigned int i=0;i<mOut.size();i++)
    {
        mOut(i)=_pActiv->derivation(outWeight(i));
    }

    return mOut;
}
///////////////////////////////////////////////////////////////////////////////
MatrixFloat& ActivationLayer::get_weight()
{
    return _weight;
}
///////////////////////////////////////////////////////////////////////////////
