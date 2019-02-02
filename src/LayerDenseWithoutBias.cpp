#include "LayerDenseWithoutBias.h"

#include <cstdlib> // for rand
#include <cmath> // for sqrt

///////////////////////////////////////////////////////////////////////////////
LayerDenseWithoutBias::LayerDenseWithoutBias(int iInSize,int iOutSize):
    Layer(iInSize,iOutSize)
{
    _weight.resize((int)_iInSize,(int)_iOutSize);
    init();
}
///////////////////////////////////////////////////////////////////////////////
LayerDenseWithoutBias::~LayerDenseWithoutBias()
{ }
///////////////////////////////////////////////////////////////////////////////
void LayerDenseWithoutBias::init()
{
    float a =4.f*sqrtf(6.f/(_iInSize+_iOutSize));
    for(unsigned int i=0;i<_weight.size();i++)
        _weight(i)=((float)rand()/(float)RAND_MAX-0.5f)*2.f*a;
}
///////////////////////////////////////////////////////////////////////////////
void LayerDenseWithoutBias::forward(const MatrixFloat& mMatIn,MatrixFloat& mMatOut) const
{
    mMatOut= mMatIn*_weight ;
}
///////////////////////////////////////////////////////////////////////////////
void LayerDenseWithoutBias::backpropagation(const MatrixFloat &mInput,const MatrixFloat &mDelta, float fLearningRate, MatrixFloat &mNewDelta)
{
    mNewDelta=_weight*mDelta; //newDelta=deriv(out)/deriv(in)*delta

    //_weight-= ( (mNewDelta.cwiseProduct(mInput.transpose()))*fLearningRate);
    _weight-= (mInput.transpose())*(mDelta.transpose())*fLearningRate; //todo optimize
}
///////////////////////////////////////////////////////////////////////////////
