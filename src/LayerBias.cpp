/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#include "LayerBias.h"
#include "Initializers.h"

///////////////////////////////////////////////////////////////////////////////
LayerBias::LayerBias(const string& sBiasInitializer) :
    Layer("Bias")
{
    set_bias_initializer(sBiasInitializer);
    LayerBias::init();
}
///////////////////////////////////////////////////////////////////////////////
LayerBias::~LayerBias()
{ }
///////////////////////////////////////////////////////////////////////////////
Layer* LayerBias::clone() const
{
    LayerBias* pLayer=new LayerBias(bias_initializer());
	pLayer->_bias = _bias;

    return pLayer;
}
///////////////////////////////////////////////////////////////////////////////
void LayerBias::init()
{
	_bias.resize(0,0);
    Layer::init();
}
///////////////////////////////////////////////////////////////////////////////
void LayerBias::forward(const MatrixFloat& mIn,MatrixFloat& mOut)
{
	if(_bias.size()==0)
        Initializers::compute(bias_initializer(), _bias, 1, mIn.cols());

    mOut = rowWiseAdd( mIn , _bias);
}
///////////////////////////////////////////////////////////////////////////////
void LayerBias::backpropagation(const MatrixFloat &mIn,const MatrixFloat &mGradientOut, MatrixFloat &mGradientIn)
{
	(void)mIn;
	
	_gradientBias = colWiseMean(mGradientOut);

	if (_bFirstLayer)
		return;

    mGradientIn = mGradientOut;
}
///////////////////////////////////////////////////////////////