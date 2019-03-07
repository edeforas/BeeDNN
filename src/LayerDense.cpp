#include "LayerDense.h"

#include <cstdlib> // for rand
#include <cmath> // for sqrt
#include "Optimizer.h"

///////////////////////////////////////////////////////////////////////////////
LayerDense::LayerDense(int iInSize, int iOutSize, bool bHasBias) :
	Layer(iInSize , iOutSize, "Dense"),
	_bHasBias(bHasBias)
{
    _weight.resize(_iInSize+(bHasBias?1:0),_iOutSize);

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

//	if (_bHasBias)
//		lastRow(_weight).setZero();

	Layer::init();
}
///////////////////////////////////////////////////////////////////////////////
void LayerDense::forward(const MatrixFloat& mMatIn,MatrixFloat& mMatOut) const
{
	if (_bHasBias)
		mMatOut = mMatIn * withoutLastRow(_weight) + lastRow(_weight);
	else
		mMatOut = mMatIn * _weight;
}
///////////////////////////////////////////////////////////////////////////////
void LayerDense::backpropagation(const MatrixFloat &mInput,const MatrixFloat &mDelta, Optimizer* pOptim, MatrixFloat &mNewDelta)
{
	if (_bHasBias)
	{
		mNewDelta = mDelta * (withoutLastRow(_weight).transpose());
		pOptim->optimize(_weight, addColumnOfOne(mInput), mDelta); //temp
	}
	else
	{
		mNewDelta = mDelta * (_weight.transpose());
		pOptim->optimize(_weight, mInput, mDelta);
	}
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
