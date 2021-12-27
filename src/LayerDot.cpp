/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#include "LayerDot.h"

#include "Initializers.h"

#include <cmath> // for sqrt

///////////////////////////////////////////////////////////////////////////////
LayerDot::LayerDot(Index iInputSize, Index iOutputSize, string sWeightInitializer) :
    Layer( "Dot"),
	_iInputSize(iInputSize),
	_iOutputSize(iOutputSize)
{
	set_weight_initializer(sWeightInitializer);
	LayerDot::init();
}
///////////////////////////////////////////////////////////////////////////////
LayerDot::~LayerDot()
{ }
///////////////////////////////////////////////////////////////////////////////
Layer* LayerDot::clone() const
{
    LayerDot* pLayer=new LayerDot(_iInputSize, _iOutputSize);
    pLayer->_weight=_weight;
	return pLayer;
}
///////////////////////////////////////////////////////////////////////////////
void LayerDot::init()
{
	if (_iInputSize == 0)
		return;

	if (_iOutputSize == 0)
		return;

	assert(_iInputSize > 0);
	assert(_iOutputSize > 0);
	
	Initializers::compute(weight_initializer(), _weight, _iInputSize, _iOutputSize);
	
	Layer::init();
}
///////////////////////////////////////////////////////////////////////////////
void LayerDot::forward(const MatrixFloat& mIn,MatrixFloat& mOut)
{
	assert(mIn.cols() == _weight.rows());
	mOut = mIn * _weight;
}
///////////////////////////////////////////////////////////////////////////////
void LayerDot::backpropagation(const MatrixFloat &mIn,const MatrixFloat &mGradientOut, MatrixFloat &mGradientIn)
{
	// average the gradient as in: https://stats.stackexchange.com/questions/183840/sum-or-average-of-gradients-in-mini-batch-gradient-decent
	_gradientWeight = (mIn.transpose())*mGradientOut*(1.f / mIn.rows());

	if (!_bFirstLayer)
		mGradientIn = mGradientOut * (_weight.transpose());
}
///////////////////////////////////////////////////////////////
Index LayerDot::input_size() const
{
	return _iInputSize;
}
///////////////////////////////////////////////////////////////
Index LayerDot::output_size() const
{
	return _iOutputSize;
}
///////////////////////////////////////////////////////////////