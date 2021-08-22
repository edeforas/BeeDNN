/*
    Copyright (c) 2021, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#include "LayerTimeDistributedDot.h"

///////////////////////////////////////////////////////////////////////////////
LayerTimeDistributedDot::LayerTimeDistributedDot(int iFrameSize, int iOutFrameSize) :
    Layer("Bias")
{
	_iFrameSize=iFrameSize;
	_iOutFrameSize=iOutFrameSize;
    LayerTimeDistributedDot::init();
}
///////////////////////////////////////////////////////////////////////////////
LayerTimeDistributedDot::~LayerTimeDistributedDot()
{ }
///////////////////////////////////////////////////////////////////////////////
Layer* LayerTimeDistributedDot::clone() const
{
    LayerTimeDistributedDot* pLayer=new LayerTimeDistributedDot(_iFrameSize,_iOutFrameSize);
	pLayer->_weight = _weight;

    return pLayer;
}
///////////////////////////////////////////////////////////////////////////////
void LayerTimeDistributedDot::init()
{
	_weight.resize(0,0);
    Layer::init();
}
///////////////////////////////////////////////////////////////////////////////
void LayerTimeDistributedDot::forward(const MatrixFloat& mIn,MatrixFloat& mOut)
{
	if(_weight.size()==0)
		_weight.setZero(1, _iFrameSize);

    mOut = rowWiseTimeDistributedAdd( mIn , _bias);
}
///////////////////////////////////////////////////////////////////////////////
void LayerTimeDistributedDot::backpropagation(const MatrixFloat &mIn,const MatrixFloat &mGradientOut, MatrixFloat &mGradientIn)
{
	(void)mIn;
	
	_gradientBias = colWiseTimeDistributedMean(mGradientOut,_iFrameSize);

	if (_bFirstLayer)
		return;

    mGradientIn = mGradientOut;
}
///////////////////////////////////////////////////////////////