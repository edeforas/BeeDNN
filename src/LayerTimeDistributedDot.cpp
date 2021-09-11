/*
    Copyright (c) 2021, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#include "LayerTimeDistributedDot.h"

///////////////////////////////////////////////////////////////////////////////
LayerTimeDistributedDot::LayerTimeDistributedDot(int iInFrameSize, int iOutFrameSize) :
    Layer("TimeDistributedDot")
{
	_iInFrameSize=iInFrameSize;
	_iOutFrameSize=iOutFrameSize;
    LayerTimeDistributedDot::init();
}
///////////////////////////////////////////////////////////////////////////////
LayerTimeDistributedDot::~LayerTimeDistributedDot()
{ }
///////////////////////////////////////////////////////////////////////////////
Layer* LayerTimeDistributedDot::clone() const
{
    LayerTimeDistributedDot* pLayer=new LayerTimeDistributedDot(_iInFrameSize,_iOutFrameSize);
	pLayer->_weight = _weight;

    return pLayer;
}
///////////////////////////////////////////////////////////////////////////////
void LayerTimeDistributedDot::init()
{
	if (_iInFrameSize == 0)
		return;

	if (_iOutFrameSize == 0)
		return;

	assert(_iInFrameSize > 0);
	assert(_iOutFrameSize > 0);

	//Xavier uniform initialization
	float a = sqrtf(6.f / (_iInFrameSize + _iOutFrameSize));
	_weight.setRandom(_iInFrameSize, _iOutFrameSize);
	_weight *= a;

    Layer::init();
}
///////////////////////////////////////////////////////////////////////////////
void LayerTimeDistributedDot::forward(const MatrixFloat& mIn,MatrixFloat& mOut)
{
	Index iNbFrames = mIn.cols() / _iInFrameSize;
	mOut = constResize(mIn,mIn.size() / _iInFrameSize,_iInFrameSize) * _weight;
	mOut.resize(mIn.rows(), _iOutFrameSize* iNbFrames);
}
///////////////////////////////////////////////////////////////////////////////
void LayerTimeDistributedDot::backpropagation(const MatrixFloat &mIn,const MatrixFloat &mGradientOut, MatrixFloat &mGradientIn)
{
	(void)mIn;

	MatrixFloat mGradOut = constResize(mGradientOut, mIn.size() / _iOutFrameSize, _iOutFrameSize);

	// average the gradient as in: https://stats.stackexchange.com/questions/183840/sum-or-average-of-gradients-in-mini-batch-gradient-decent
	_gradientWeight = (mIn.transpose()) * mGradOut * (1.f / mIn.rows());

	if (!_bFirstLayer)
	{
		Index iNbFrames = mGradientOut.cols() / _iOutFrameSize;
		mGradientIn = mGradOut * (_weight.transpose());
		mGradientIn.resize(mGradientOut.rows(), _iInFrameSize* iNbFrames);
	}
}
///////////////////////////////////////////////////////////////