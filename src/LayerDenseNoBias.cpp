#include "LayerDenseNoBias.h"

#include <cstdlib> // for rand
#include <cmath> // for sqrt

#include "MatrixUtil.h"

///////////////////////////////////////////////////////////////////////////////
LayerDenseNoBias::LayerDenseNoBias(int iInSize,int iOutSize):
    Layer(iInSize,iOutSize)
{
    _weight.resize((int)_iInSize,(int)_iOutSize);
    init();
}
///////////////////////////////////////////////////////////////////////////////
LayerDenseNoBias::~LayerDenseNoBias()
{ }
///////////////////////////////////////////////////////////////////////////////
void LayerDenseNoBias::init()
{
    float a =4.f*sqrtf(6.f/(_iInSize+_iOutSize));
    for(int i=0;i<_weight.size();i++)
        _weight(i)=((float)rand()/(float)RAND_MAX-0.5f)*2.f*a;
}
///////////////////////////////////////////////////////////////////////////////
void LayerDenseNoBias::forward(const MatrixFloat& mMatIn,MatrixFloat& mMatOut) const
{
    mMatOut= mMatIn*_weight;
}
///////////////////////////////////////////////////////////////////////////////
void LayerDenseNoBias::backpropagation(const MatrixFloat &mInput,const MatrixFloat &mDelta, float fLearningRate, MatrixFloat &mNewDelta)
{
    mNewDelta=_weight*mDelta;
    _weight-= (mDelta*mInput).transpose()*fLearningRate; //todo optimize
}
///////////////////////////////////////////////////////////////////////////////
void LayerDenseNoBias::to_string(string& sBuffer)
{
    sBuffer+="DenseNoBias:  InSize: "+std::to_string(_iInSize) +" OutSize: "+std::to_string(_iOutSize)+"\n";
    sBuffer+="Weight:\n";
    sBuffer+=MatrixUtil::to_string(_weight);
}
///////////////////////////////////////////////////////////////////////////////
