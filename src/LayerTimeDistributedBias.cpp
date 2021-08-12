/*
    Copyright (c) 2021, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#include "LayerTimeDistributedBias.h"

///////////////////////////////////////////////////////////////////////////////
LayerTimeDistributedBias::LayerTimeDistributedBias(int iFrameSize) :
    Layer("Bias")
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
void LayerTimeDistributedBias::init()
{
	_bias.resize(0,0);
    Layer::init();
}
///////////////////////////////////////////////////////////////////////////////
void LayerTimeDistributedBias::forward(const MatrixFloat& mIn,MatrixFloat& mOut)
{
	if(_bias.size()==0)
		_bias.setZero(1, _iFrameSize);

    mOut = rowWiseTimeDistributedAdd( mIn , _bias);
}
///////////////////////////////////////////////////////////////////////////////
void LayerTimeDistributedBias::backpropagation(const MatrixFloat &mIn,const MatrixFloat &mGradientOut, MatrixFloat &mGradientIn)
{
	(void)mIn;
	
	_gradientBias = colWiseTimeDistributedMean(mGradientOut,_iFrameSize);

	if (_bFirstLayer)
		return;

    mGradientIn = mGradientOut;
}
///////////////////////////////////////////////////////////////