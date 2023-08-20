/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#include "LayerDense.h"
#include "Initializers.h"
namespace bee {

///////////////////////////////////////////////////////////////////////////////
LayerDense::LayerDense(Index iInputSize, Index iOutputSize, const string& sWeightInitializer, const string& sBiasInitializer) :
    Layer( "Dense"),
	_iInputSize(iInputSize),
	_iOutputSize(iOutputSize)
{
	set_weight_initializer(sWeightInitializer);
	set_bias_initializer(sBiasInitializer);
	LayerDense::init();
}
///////////////////////////////////////////////////////////////////////////////
LayerDense::~LayerDense()
{ }
///////////////////////////////////////////////////////////////////////////////
Layer* LayerDense::clone() const
{
    LayerDense* pLayer=new LayerDense(_iInputSize, _iOutputSize,weight_initializer(),bias_initializer());
    pLayer->_weight = _weight;
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

    // init weights and bias
	Initializers::compute(weight_initializer(),_weight, _iInputSize, _iOutputSize);
	Initializers::compute(bias_initializer(),_bias, 1, _iOutputSize);
	
	Layer::init();
}
///////////////////////////////////////////////////////////////////////////////
void LayerDense::forward(const MatrixFloat& mIn,MatrixFloat& mOut)
{
	assert(mIn.cols() == _weight.rows());
	assert(_weight.cols() == _bias.cols());
	mOut = mIn * _weight;
    mOut = rowWiseAdd(mOut,_bias);
}
///////////////////////////////////////////////////////////////////////////////
void LayerDense::backpropagation(const MatrixFloat &mIn,const MatrixFloat &mGradientOut, MatrixFloat &mGradientIn)
{
	_gradientWeight = mIn.transpose()*mGradientOut;

	// average the gradient as in: https://stats.stackexchange.com/questions/183840/sum-or-average-of-gradients-in-mini-batch-gradient-decent
	_gradientWeight *= (1.f / mIn.rows());

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

}