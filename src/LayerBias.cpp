/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#include "LayerBias.h"

///////////////////////////////////////////////////////////////////////////////
LayerBias::LayerBias() :
    Layer(0, 0, "Bias")
{
    LayerBias::init();
}
///////////////////////////////////////////////////////////////////////////////
LayerBias::~LayerBias()
{ }
///////////////////////////////////////////////////////////////////////////////
Layer* LayerBias::clone() const
{
    LayerBias* pLayer=new LayerBias();
	pLayer->weights() = _weight;

    return pLayer;
}
///////////////////////////////////////////////////////////////////////////////
void LayerBias::init()
{
    _weight.resize(0,0);
    Layer::init();
}
///////////////////////////////////////////////////////////////////////////////
void LayerBias::forward(const MatrixFloat& mIn,MatrixFloat& mOut)
{
	if(_weight.size()==0)
		_weight.setZero(1, mIn.cols());

    mOut = rowWiseAdd( mIn/*.array()*/ , _weight);
}
///////////////////////////////////////////////////////////////////////////////
void LayerBias::backpropagation(const MatrixFloat &mIn,const MatrixFloat &mGradientOut, MatrixFloat &mGradientIn)
{
	(void)mIn;
	
	_gradientWeight = colWiseMean(mGradientOut);

	if (_bFirstLayer)
		return;

    mGradientIn = mGradientOut;
}
///////////////////////////////////////////////////////////////