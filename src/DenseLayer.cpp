#include "DenseLayer.h"

#include <cassert>
#include <cstdlib> // for rand
#include <cmath> // for sqrt

#include "Activation.h"

///////////////////////////////////////////////////////////////////////////////
DenseLayer::DenseLayer(int iInSize,int iOutSize,const Activation* activ): Layer(),
    _activ(activ),
    _iInSize(iInSize),
    _iOutSize(iOutSize)
{
    assert(activ);
    _weight.resize(_iInSize+1,_iOutSize); //+1 for bias
}
///////////////////////////////////////////////////////////////////////////////
DenseLayer::~DenseLayer()
{ }
///////////////////////////////////////////////////////////////////////////////
void DenseLayer::init()
{
    double a =sqrt(6./(_iInSize+_iOutSize));

    if(_activ->name()=="Sigmoid")
        a*=4.;

    for(unsigned int i=0;i<_weight.size();i++)
    {
        _weight(i)=((double)rand()/(double)RAND_MAX-0.5)*2.*a;
    }

    dE.resize(_iInSize+1,_iOutSize);
    dE.set_zero();
}
///////////////////////////////////////////////////////////////////////////////
void DenseLayer::forward(const Matrix& mMatIn,Matrix& mMatOut) const
{
    // compute out=[in 1]*weight; todo use MAC
    mMatOut=mMatIn*(_weight.without_last_row());
    mMatOut+=_weight.row(_iInSize);

    // apply activation
    for(unsigned int i=0;i<mMatOut.size();i++)
    {
        mMatOut(i)=_activ->apply(mMatOut(i)); //todo keep matrix in layer, do not resize
    }
}
///////////////////////////////////////////////////////////////////////////////
void DenseLayer::forward_save(const Matrix& mMatIn,Matrix& mMatOut)
{
    in=mMatIn;
    // compute out=[in 1]*weight; todo use MAC
    mMatOut=mMatIn*_weight.without_last_row();
    mMatOut+=_weight.row(_iInSize);

    outWeight=mMatOut;

    // apply activation
    for(unsigned int i=0;i<mMatOut.size();i++)
    {
        mMatOut(i)=_activ->apply(mMatOut(i)); //todo keep matrix in layer, do not resize
    }
    out=mMatOut;
}
///////////////////////////////////////////////////////////////////////////////
Matrix DenseLayer::get_weight_activation_derivation()
{
    // apply activation derivation on outweight
    Matrix mOut=outWeight;
    for(unsigned int i=0;i<mOut.size();i++)
    {
        mOut(i)=_activ->derivation(outWeight(i),out(i));
    }

    return mOut;
}
///////////////////////////////////////////////////////////////////////////////
Matrix& DenseLayer::get_weight()
{
    return _weight;
}
///////////////////////////////////////////////////////////////////////////////
