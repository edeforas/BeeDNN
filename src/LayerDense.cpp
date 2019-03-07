#include "LayerDense.h"

#include <cstdlib> // for rand
#include <cmath> // for sqrt
#include "Optimizer.h"

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
		_bias.setZero();

		_mOne.resize(1, 1);
		_mOne.setConstant(1.f);
	}

	Layer::init();
}
///////////////////////////////////////////////////////////////////////////////
void LayerDense::forward(const MatrixFloat& mMatIn,MatrixFloat& mMatOut) const
{
	mMatOut = mMatIn * _weight;

	if (_bHasBias)
		mMatOut +=  _bias;
}
///////////////////////////////////////////////////////////////////////////////
void LayerDense::backpropagation(const MatrixFloat &mInput, const MatrixFloat &mDelta, Optimizer* pOptim, MatrixFloat &mNewDelta)
{
	mNewDelta = mDelta * (_weight.transpose());

	if (_bHasBias)
	{ 
		//concat weight and bias , optimize, and then split, todo replace with better computation
		contatenateHorizontallyInto(mInput, _mOne, _mInAndOne);
		contatenateVerticallyInto(_weight, _bias, _fullWeight);

        _mDx = (_mInAndOne.transpose())*mDelta;

		pOptim->optimize(_fullWeight, _mDx);

		_weight = withoutLastRow(_fullWeight);
        _bias = _fullWeight.row(_fullWeight.rows() - 1);
	}
	else
	{
		_mDx = mInput.transpose()*mDelta;
		pOptim->optimize(_weight, _mDx);
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
