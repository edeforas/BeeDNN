/*
    Copyright (c) 2021, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#include "LayerTimeDistributedDense.h"
#include "Initializers.h"

using namespace std;
namespace beednn {

///////////////////////////////////////////////////////////////////////////////
LayerTimeDistributedDense::LayerTimeDistributedDense(int iInFrameSize, int iOutFrameSize, const string& sWeightInitializer, const string& sBiasInitializer) :
    Layer("TimeDistributedDense")
{
	_iInFrameSize=iInFrameSize;
	_iOutFrameSize=iOutFrameSize;

	set_weight_initializer(sWeightInitializer);
	set_bias_initializer(sBiasInitializer);

    LayerTimeDistributedDense::init();
}
///////////////////////////////////////////////////////////////////////////////
LayerTimeDistributedDense::~LayerTimeDistributedDense()
{ }
///////////////////////////////////////////////////////////////////////////////
Layer* LayerTimeDistributedDense::clone() const
{
    LayerTimeDistributedDense* pLayer=new LayerTimeDistributedDense(_iInFrameSize,_iOutFrameSize, weight_initializer(), bias_initializer());
	pLayer->_weight = _weight;
	pLayer->_bias = _bias;

    return pLayer;
}
///////////////////////////////////////////////////////////////////////////////
int LayerTimeDistributedDense::in_frame_size() const
{
	return _iInFrameSize;
}
///////////////////////////////////////////////////////////////////////////////
int LayerTimeDistributedDense::out_frame_size() const
{
	return _iOutFrameSize;
}
///////////////////////////////////////////////////////////////////////////////
void LayerTimeDistributedDense::init()
{
	Initializers::compute(weight_initializer(), _weight, _iInFrameSize, _iOutFrameSize);
	Initializers::compute(bias_initializer(), _bias, 1, _iOutFrameSize);

    Layer::init();
}
///////////////////////////////////////////////////////////////////////////////
void LayerTimeDistributedDense::forward(const MatrixFloat& mIn,MatrixFloat& mOut)
{
	// reshape the input to (x, _iFrameSize), compute, reshape back
	Index iNbFrames = mIn.cols() / _iInFrameSize;
	MatrixFloat mInR = viewResize(mIn, iNbFrames* mIn.rows(), _iInFrameSize);
	mOut = mInR * _weight;
	
    mOut=rowWiseAdd(mOut, _bias);
	mOut.resize(mIn.rows(), iNbFrames*_iOutFrameSize);
}
///////////////////////////////////////////////////////////////////////////////
void LayerTimeDistributedDense::backpropagation(const MatrixFloat &mIn,const MatrixFloat &mGradientOut, MatrixFloat &mGradientIn)
{
	// average the gradient as in: https://stats.stackexchange.com/questions/183840/sum-or-average-of-gradients-in-mini-batch-gradient-decent

	// reshape the input and gradient to (x, _iFrameSize), compute product, reshape back
	Index iNbFrames = mGradientOut.cols() / _iOutFrameSize;
	MatrixFloat mGradientOutR = viewResize(mGradientOut, iNbFrames * mGradientOut.rows(), _iOutFrameSize);
	MatrixFloat mInR = viewResize(mIn, iNbFrames * mIn.rows(), _iInFrameSize);
	
	_gradientWeight = (mInR.transpose()) * mGradientOutR * (1.f / mIn.rows());
    _gradientBias = colWiseMean(mGradientOutR);

	if (!_bFirstLayer)
	{
		mGradientIn = mGradientOutR * (_weight.transpose());
		mGradientIn.resize(mIn.rows(), iNbFrames * _iInFrameSize);
	}
}
///////////////////////////////////////////////////////////////
}