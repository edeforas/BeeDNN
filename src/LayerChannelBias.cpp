/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#include "LayerChannelBias.h"

///////////////////////////////////////////////////////////////////////////////
LayerChannelBias::LayerChannelBias(Index iNbRows,Index iNbCols,Index iNbChannels) :
    Layer(0, 0, "ChannelBias")
{
	_iNbRows=iNbRows;
	_iNbCols=iNbCols;
	_iNbChannels=iNbChannels;
	
    _weight.resize(_iNbChannels,1);
	_gradientWeight.resize(_iNbChannels, 1);
    LayerChannelBias::init();
}
///////////////////////////////////////////////////////////////////////////////
LayerChannelBias::~LayerChannelBias()
{ }
///////////////////////////////////////////////////////////////////////////////
Layer* LayerChannelBias::clone() const
{
    LayerChannelBias* pLayer=new LayerChannelBias(_iNbRows,_iNbCols,_iNbChannels);
	pLayer->weights() = _weight;
	return pLayer;
}
///////////////////////////////////////////////////////////////////////////////
void LayerChannelBias::init()
{
    _weight.setZero();
    Layer::init();
}
///////////////////////////////////////////////////////////////////////////////
void LayerChannelBias::forward(const MatrixFloat& mIn,MatrixFloat& mOut)
{
    mOut = mIn.array() + _weight(0);
}
///////////////////////////////////////////////////////////////////////////////
void LayerChannelBias::backpropagation(const MatrixFloat &mIn,const MatrixFloat &mGradientOut, MatrixFloat &mGradientIn)
{
	(void)mIn;

	_gradientWeight(0) = mGradientOut.mean();
	
	if (_bFirstLayer)
		return;

    mGradientIn = mGradientOut;
}
///////////////////////////////////////////////////////////////////////////////