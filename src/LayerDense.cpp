/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#include "LayerDense.h"

#include <cmath> // for sqrt

///////////////////////////////////////////////////////////////////////////////
LayerDense::LayerDense(Index iInputSize, Index iOutputSize, bool bHasBias) :
    Layer( "Dense"),
    _bHasBias(bHasBias)
{
	_iInputSize = iInputSize;
	_iOutputSize = iOutputSize;
    LayerDense::init();
}
///////////////////////////////////////////////////////////////////////////////
LayerDense::~LayerDense()
{ }
///////////////////////////////////////////////////////////////////////////////
Layer* LayerDense::clone() const
{
    LayerDense* pLayer=new LayerDense(_iInputSize, _iOutputSize,_bHasBias);
    pLayer->_weight=_weight;
    return pLayer;
}
///////////////////////////////////////////////////////////////////////////////
void LayerDense::init()
{
	assert(_iInputSize > 0);
	assert(_iOutputSize > 0);
	
	_weight.resize(_iInputSize +(_bHasBias?1:0), _iOutputSize);

    //Xavier uniform initialization
    float a =sqrtf(6.f/(_iInputSize + _iOutputSize));
    _weight.setRandom();
    _weight*=a;

    if (_bHasBias)
        _weight.row(_iInputSize).setZero();

    Layer::init();
}
///////////////////////////////////////////////////////////////////////////////
void LayerDense::forward(const MatrixFloat& mIn,MatrixFloat& mOut)
{
    if (_bHasBias)
        mOut = rowWiseAdd(mIn *_weight.topRows(_iInputSize) , _weight.row(_iInputSize)); //split _weight in [weightnobias, bias] in computation
    else
        mOut = mIn * _weight; //todo use W*x instead of x*W ?
}
///////////////////////////////////////////////////////////////////////////////
void LayerDense::backpropagation(const MatrixFloat &mIn,const MatrixFloat &mGradientOut, MatrixFloat &mGradientIn)
{
    if (_bHasBias)
    {
		// optimisation: split _weight in [weightnobias, bias]
		_gradientWeight = ((addColumnOfOne(mIn)).transpose())*mGradientOut*(1.f / mIn.rows()); //todo optimize
        
		if (_bFirstLayer)
			return;

		mGradientIn = mGradientOut * _weight.topRows(_iInputSize).transpose();
    }
    else
    {
		// average the gradient as in:
		// https://stats.stackexchange.com/questions/183840/sum-or-average-of-gradients-in-mini-batch-gradient-decent
        _gradientWeight = (mIn.transpose())*mGradientOut*(1.f / mIn.rows());

		if (_bFirstLayer)
			return;

		mGradientIn = mGradientOut * (_weight.transpose());
    }
}
///////////////////////////////////////////////////////////////
bool LayerDense::has_bias() const
{
    return _bHasBias;
}
///////////////////////////////////////////////////////////////
Index LayerDense::input_size() const
{
	return _iInputSize;
}
///////////////////////////////////////////////////////////////
Index LayerDense::output_size() const
{
	return _iOutputSize;
}
///////////////////////////////////////////////////////////////