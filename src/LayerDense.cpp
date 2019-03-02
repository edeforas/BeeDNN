#include "LayerDense.h"

#include <cstdlib> // for rand
#include <cmath> // for sqrt

///////////////////////////////////////////////////////////////////////////////
LayerDense::LayerDense(int iInSize,int iOutSize,bool bHasBias):
    Layer(iInSize,iOutSize,"Dense"),
	_bHasBias(bHasBias)
{
    _weight.resize(_iInSize,_iOutSize);

	if(_bHasBias)
		_bias.resize(1,_iOutSize);

    LayerDense::init();
}
///////////////////////////////////////////////////////////////////////////////
LayerDense::~LayerDense()
{ }
///////////////////////////////////////////////////////////////////////////////
void LayerDense::init()
{
    //Xavier uniform initialization
    float a =sqrtf(6.f/(_iInSize+_iOutSize));
    for(int i=0;i<_weight.size();i++)
        _weight(i)=((float)rand()/(float)RAND_MAX-0.5f)*2.f*a;

	if (_bHasBias)
	{
		for (int i = 0; i < _bias.size(); i++)
			_bias(i) = 0.f;
	}

	Layer::init();
}
///////////////////////////////////////////////////////////////////////////////
void LayerDense::forward(const MatrixFloat& mMatIn,MatrixFloat& mMatOut) const
{
	if (_bHasBias)
		mMatOut = mMatIn*_weight+_bias;
	else
		mMatOut = mMatIn*_weight;
}
///////////////////////////////////////////////////////////////////////////////
void LayerDense::backpropagation(const MatrixFloat &mInput,const MatrixFloat &mDelta, float fLearningRate, MatrixFloat &mNewDelta)
{
    mNewDelta=mDelta*(_weight.transpose());

    _weight-=(mInput.transpose())*(mDelta*fLearningRate);
	
	if (_bHasBias)
		_bias-=mDelta*fLearningRate;
}
///////////////////////////////////////////////////////////////////////////////
const MatrixFloat& LayerDense::weight() const
{
    return _weight;
}
///////////////////////////////////////////////////////////////////////////////
const MatrixFloat& LayerDense::bias() const
{
    return _bias;
}
///////////////////////////////////////////////////////////////////////////////
bool LayerDense::has_bias() const
{
	return _bHasBias;
}
///////////////////////////////////////////////////////////////////////////////
