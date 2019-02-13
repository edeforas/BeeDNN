#include "LayerDenseAndBias.h"

#include <cstdlib> // for rand
#include <cmath> // for sqrt

//#include "MatrixUtil.h"

///////////////////////////////////////////////////////////////////////////////
LayerDenseAndBias::LayerDenseAndBias(int iInSize,int iOutSize):
    Layer(iInSize,iOutSize,"DenseAndBias")
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
    //Xavier uniform initialisation
    float a =sqrtf(6.f/(_iInSize+_iOutSize));
    for(int i=0;i<_weight.size();i++)
        _weight(i)=((float)rand()/(float)RAND_MAX-0.5f)*2.f*a;

    for(int i=0;i<_bias.size();i++)
        _bias(i)=0.f;
}
///////////////////////////////////////////////////////////////////////////////
void LayerDenseAndBias::forward(const MatrixFloat& mMatIn,MatrixFloat& mMatOut) const
{
    mMatOut= mMatIn*_weight+_bias;
}
///////////////////////////////////////////////////////////////////////////////
void LayerDenseAndBias::backpropagation(const MatrixFloat &mInput,const MatrixFloat &mDelta, float fLearningRate, MatrixFloat &mNewDelta)
{
    mNewDelta=_weight*mDelta;

    _weight-=((mDelta*mInput).transpose())*fLearningRate;
    _bias-=(mDelta.transpose())*fLearningRate;
}
///////////////////////////////////////////////////////////////////////////////
const MatrixFloat& LayerDenseAndBias::weight() const
{
    return _weight;
}
///////////////////////////////////////////////////////////////////////////////
const MatrixFloat& LayerDenseAndBias::bias() const
{
    return _bias;
}
///////////////////////////////////////////////////////////////////////////////
