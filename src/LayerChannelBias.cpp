/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#include "LayerChannelBias.h"
#include "Initializers.h"
namespace bee {

///////////////////////////////////////////////////////////////////////////////
LayerChannelBias::LayerChannelBias(Index iNbRows,Index iNbCols,Index iNbChannels, const string& sBiasInitializer) :
    Layer("ChannelBias")
{
	_iNbRows=iNbRows;
	_iNbCols=iNbCols;
	_iNbChannels=iNbChannels;

	set_bias_initializer(sBiasInitializer);
    LayerChannelBias::init();
}
///////////////////////////////////////////////////////////////////////////////
LayerChannelBias::~LayerChannelBias()
{ }
///////////////////////////////////////////////////////////////////////////////
Layer* LayerChannelBias::clone() const
{
    LayerChannelBias* pLayer=new LayerChannelBias(_iNbRows,_iNbCols,_iNbChannels, bias_initializer());
	pLayer->_bias = _bias;
	return pLayer;
}
///////////////////////////////////////////////////////////////////////////////
void LayerChannelBias::init()
{
	Initializers::compute(bias_initializer(), _bias, 1, _iNbChannels);
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
void LayerChannelBias::forward(const MatrixFloat& mIn,MatrixFloat& mOut)
{
	mOut = mIn;
	channelWiseAdd(mOut, mIn.rows(),_iNbChannels,_iNbRows,_iNbCols, _bias);
}
///////////////////////////////////////////////////////////////////////////////
void LayerChannelBias::backpropagation(const MatrixFloat &mIn,const MatrixFloat &mGradientOut, MatrixFloat &mGradientIn)
{
	(void)mIn;

	_gradientBias =channelWiseMean(mGradientOut, mGradientOut.rows(),_iNbChannels,_iNbRows,_iNbCols);

	if (_bFirstLayer)
		return;

    mGradientIn = mGradientOut;
}
///////////////////////////////////////////////////////////////////////////////
}