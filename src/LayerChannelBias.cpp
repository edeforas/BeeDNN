/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#include "LayerChannelBias.h"

///////////////////////////////////////////////////////////////////////////////
LayerChannelBias::LayerChannelBias(Index iNbRows,Index iNbCols,Index iNbChannels) :
    Layer("ChannelBias")
{
	_iNbRows=iNbRows;
	_iNbCols=iNbCols;
	_iNbChannels=iNbChannels;
	
    _bias.setZero(1,_iNbChannels);
	_gradientBias.setZero(1,_iNbChannels);
    LayerChannelBias::init();
}
///////////////////////////////////////////////////////////////////////////////
LayerChannelBias::~LayerChannelBias()
{ }
///////////////////////////////////////////////////////////////////////////////
Layer* LayerChannelBias::clone() const
{
    LayerChannelBias* pLayer=new LayerChannelBias(_iNbRows,_iNbCols,_iNbChannels);
	pLayer->_bias = _bias;
	return pLayer;
}
///////////////////////////////////////////////////////////////////////////////
void LayerChannelBias::init()
{
	_bias.setZero();
    Layer::init();
}
///////////////////////////////////////////////////////////////////////////////
void LayerChannelBias::get_params(Index & iRows, Index & iCols, Index & iChannels) const
{
	iRows = _iNbRows;
	iCols = _iNbCols;
	iChannels = _iNbChannels;
}
///////////////////////////////////////////////////////////////////////////////
void LayerChannelBias::predict(const MatrixFloat& mIn,MatrixFloat& mOut)
{
	mOut = mIn;
	Index iNbSamples = mIn.rows();
	channelWiseAdd(mOut,iNbSamples,_iNbChannels,_iNbRows,_iNbCols, _bias);
}
///////////////////////////////////////////////////////////////////////////////
void LayerChannelBias::backpropagation(const MatrixFloat &mIn,const MatrixFloat &mGradientOut, MatrixFloat &mGradientIn)
{
	(void)mIn;

	Index iNbSamples = mGradientOut.rows();
	_gradientBias =channelWiseMean(mGradientOut,iNbSamples,_iNbChannels,_iNbRows,_iNbCols);

	if (_bFirstLayer)
		return;

    mGradientIn = mGradientOut;
}
///////////////////////////////////////////////////////////////////////////////