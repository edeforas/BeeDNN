#include "LayerDenseAndBias.h"

#include <cstdlib> // for rand
#include <cmath> // for sqrt

///////////////////////////////////////////////////////////////////////////////
LayerDenseAndBias::LayerDenseAndBias(int iInSize,int iOutSize):
    Layer(iInSize,iOutSize)
{
    _weight.resize(_iInSize,_iOutSize);
    _bias.resize(1,_iOutSize);
    init();
}
///////////////////////////////////////////////////////////////////////////////
LayerDenseAndBias::~LayerDenseAndBias()
{ }
///////////////////////////////////////////////////////////////////////////////
void LayerDenseAndBias::init()
{
    float a =4.f*sqrtf(6.f/(_iInSize+_iOutSize));

    for(unsigned int i=0;i<_weight.size();i++)
        _weight(i)=((float)rand()/(float)RAND_MAX-0.5f)*2.f*a;
	
    for(unsigned int i=0;i<_bias.size();i++)
        _bias(i)=((float)rand()/(float)RAND_MAX-0.5f)*2.f*a;
}
///////////////////////////////////////////////////////////////////////////////
void LayerDenseAndBias::forward(const MatrixFloat& mMatIn,MatrixFloat& mMatOut) const
{
    mMatOut= mMatIn*_weight + _bias;
}
///////////////////////////////////////////////////////////////////////////////
void LayerDenseAndBias::backpropagation(const MatrixFloat &mInput,const MatrixFloat &mDelta, float fLearningRate, MatrixFloat &mNewDelta)
{
    mNewDelta=_weight*mDelta; //todo BUG BUG BUG

    _weight-=(mInput.transpose())*(mDelta.transpose())*fLearningRate;  //todo optimize
    _bias-=(mDelta.transpose())*fLearningRate;
}
///////////////////////////////////////////////////////////////////////////////
