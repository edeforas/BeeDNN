/*
    Copyright (c) 2021, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#include "LayerTimeDistributedBias.h"

///////////////////////////////////////////////////////////////////////////////
LayerTimeDistributedBias::LayerTimeDistributedBias(int iFrameSize) :
    Layer("TimeDistributedBias")
{
	_iFrameSize=iFrameSize;
    LayerTimeDistributedBias::init();
}
///////////////////////////////////////////////////////////////////////////////
LayerTimeDistributedBias::~LayerTimeDistributedBias()
{ }
///////////////////////////////////////////////////////////////////////////////
Layer* LayerTimeDistributedBias::clone() const
{
    LayerTimeDistributedBias* pLayer=new LayerTimeDistributedBias(this->_iFrameSize);
	pLayer->_bias = _bias;

    return pLayer;
}
///////////////////////////////////////////////////////////////////////////////
int LayerTimeDistributedBias::frame_size() const
{
    return _iFrameSize;
}
///////////////////////////////////////////////////////////////////////////////
void LayerTimeDistributedBias::init()
{
    _bias.setZero(1, _iFrameSize);
    Layer::init();
}
///////////////////////////////////////////////////////////////////////////////
void LayerTimeDistributedBias::forward(const MatrixFloat& mIn,MatrixFloat& mOut)
{
    // reshape the input to (x, _iFrameSize), compute, reshape back
    MatrixFloat mInR = viewResize(mIn,mIn.size()/ _iFrameSize,_iFrameSize);
    mOut=rowWiseAdd(mInR, _bias);
    mOut.resize(mIn.rows(), mIn.cols());
}
///////////////////////////////////////////////////////////////////////////////
void LayerTimeDistributedBias::backpropagation(const MatrixFloat &mIn,const MatrixFloat &mGradientOut, MatrixFloat &mGradientIn)
{
    (void)mIn;

    // reshape the gradient to (x, _iFrameSize), compute
    MatrixFloat mGradientOutR = viewResize(mGradientOut, mGradientOut.size() / _iFrameSize, _iFrameSize);
    _gradientBias = colWiseMean(mGradientOutR);

	if (_bFirstLayer)
		return;

    mGradientIn = mGradientOut;
}
///////////////////////////////////////////////////////////////////////////////