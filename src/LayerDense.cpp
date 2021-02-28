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
    _bHasBias(bHasBias),
	_iInputSize(iInputSize),
	_iOutputSize(iOutputSize)
{
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
	pLayer->_bias = _bias;
	return pLayer;
}
///////////////////////////////////////////////////////////////////////////////
void LayerDense::init()
{
	if (_iInputSize == 0)
		return;

	if (_iOutputSize == 0)
		return;

	assert(_iInputSize > 0);
	assert(_iOutputSize > 0);
	
	_weight.resize(_iInputSize , _iOutputSize);

    //Xavier uniform initialization
    float a =sqrtf(6.f/(_iInputSize + _iOutputSize));
    _weight.setRandom();
    _weight*=a;

	if (_bHasBias)
		_bias.setZero(1, _iOutputSize);
	
	Layer::init();
}
///////////////////////////////////////////////////////////////////////////////
void LayerDense::forward(const MatrixFloat& mIn,MatrixFloat& mOut)
{
	mOut = mIn * _weight;

    if (_bHasBias)
        mOut = rowWiseAdd(mOut,_bias);
}
///////////////////////////////////////////////////////////////////////////////
void LayerDense::backpropagation(const MatrixFloat &mIn,const MatrixFloat &mGradientOut, MatrixFloat &mGradientIn)
{
	// average the gradient as in: https://stats.stackexchange.com/questions/183840/sum-or-average-of-gradients-in-mini-batch-gradient-decent
	_gradientWeight = (mIn.transpose())*mGradientOut*(1.f / mIn.rows());

	//TODO put in regularizer, this code force weights to be in [-1;1] using a tanh -> 1-t*t in backprop
	//_gradientWeight= _gradientWeight- _gradientWeight.cwiseProduct(_weight.cwiseProduct(_weight));

	if (_bHasBias)
		_gradientBias = colWiseMean(mGradientOut);

	if (!_bFirstLayer)
		mGradientIn = mGradientOut * (_weight.transpose());
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