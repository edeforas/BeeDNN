#include "SoftmaxLayer.h"

#include <cassert>
#include <cmath> // for exp and log

///////////////////////////////////////////////////////////////////////////////
SoftmaxLayer::SoftmaxLayer(): Layer()
{ }
///////////////////////////////////////////////////////////////////////////////
SoftmaxLayer::~SoftmaxLayer()
{ }
///////////////////////////////////////////////////////////////////////////////
void SoftmaxLayer::init()
{
//    dE.resize(_iInSize+1,_iOutSize);
//    dE.set_zero();
}
///////////////////////////////////////////////////////////////////////////////
void SoftmaxLayer::forward(const Matrix& mMatIn,Matrix& mMatOut) const
{
	//renormalisation
	double dMax=mMatIn.max();
	mMatOut=mMatIn-dMax;
	
    for(unsigned int i=0;i<mMatOut.size(); i++)
	{
		mMatOut(i)=exp(mMatOut(i));
	}

	double dSum=mMatOut.sum();
	mMatOut/=dSum;
}
///////////////////////////////////////////////////////////////////////////////
void SoftmaxLayer::backward(const Matrix& mErrorIn,Matrix& mErrorOut)
{
    // from http://cs231n.github.io/neural-networks-case-study/#grad



}
///////////////////////////////////////////////////////////////////////////////
void SoftmaxLayer::forward_save(const Matrix& mMatIn,Matrix& mMatOut)
{
		mMatOut=mMatIn;
	/*
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
	*/
}
///////////////////////////////////////////////////////////////////////////////
Matrix SoftmaxLayer::get_weight_activation_derivation()
{
	/*
	
    // apply activation derivation on outweight
    Matrix mOut=outWeight;
    for(unsigned int i=0;i<mOut.size();i++)
    {
        mOut(i)=_activ->derivation(outWeight(i),out(i));
    }

    return mOut;
*/
	}
///////////////////////////////////////////////////////////////////////////////
Matrix& SoftmaxLayer::get_weight()
{
  //  return _weight;
}
///////////////////////////////////////////////////////////////////////////////
