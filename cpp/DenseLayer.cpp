#include "DenseLayer.h"

#include <cstdlib> // for rand
#include <cmath> // for sqrt
#include "Activation.h"

///////////////////////////////////////////////////////////////////////////////
DenseLayer::DenseLayer(int iInSize,int iOutSize,const Activation& activ): Layer(),
    _activ(activ),
    _iInSize(iInSize),
    _iOutSize(iOutSize)
{
    _weight.resize(_iInSize+1,_iOutSize); //+1 for bias
}
///////////////////////////////////////////////////////////////////////////////
void DenseLayer::forward(const Matrix& mMatIn,Matrix& mMatOut) const
{
    // compute out=[in 1]*weight; todo use MAC
    mMatOut=mMatIn*(_weight.without_last_row());
    mMatOut=mMatOut+_weight.row(_iInSize);

    // apply activation
    for(int i=0;i<mMatOut.size();i++)
    {
        mMatOut(i)=_activ.forward(mMatOut(i)); //todo keep matrix in layer, do not resize
    }
}
///////////////////////////////////////////////////////////////////////////////
void DenseLayer::forward_feed(const Matrix& mMatIn,Matrix& mMatOut)
{
    in=mMatIn;
    // compute out=[in 1]*weight; todo use MAC
    mMatOut=mMatIn*_weight.without_last_row();
    mMatOut=mMatOut+_weight.row(_iInSize);

    outWeight=mMatOut;

    // apply activation
    for(int i=0;i<mMatOut.size();i++)
    {
        mMatOut(i)=_activ.forward(mMatOut(i)); //todo keep matrix in layer, do not resize
    }
    out=mMatOut;
}
///////////////////////////////////////////////////////////////////////////////
void DenseLayer::init_weight()
{
    double a =sqrt(6./(_iInSize+_iOutSize));

    if(_activ.name()=="sigmoid")
        a*=4.;

    for(int i=0;i<_weight.size();i++)
    {
        _weight(i)=((double)rand()/RAND_MAX-0.5)*2.*a;
    }
}

void DenseLayer::init_DE()
{
    dE.resize(_iInSize+1,_iOutSize);
    dE.setZero();
}

Matrix DenseLayer::get_weight_activation_derivation()
{
    // apply activation derivation on outweight
    Matrix mOut=outWeight;
    for(int i=0;i<mOut.size();i++)
    {
        mOut(i)=_activ.backward(outWeight(i),out(i));
    }

    return mOut;
}

Matrix& DenseLayer::get_weight()
{
    return _weight;
}
