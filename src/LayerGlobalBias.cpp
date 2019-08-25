/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#include "LayerGlobalBias.h"

///////////////////////////////////////////////////////////////////////////////
LayerGlobalBias::LayerGlobalBias(int inSize) :
    Layer(inSize, inSize, "GlobalBias")
{
    _weight.resize(1,1);
    LayerGlobalBias::init();
}
///////////////////////////////////////////////////////////////////////////////
LayerGlobalBias::~LayerGlobalBias()
{ }
///////////////////////////////////////////////////////////////////////////////
Layer* LayerGlobalBias::clone() const
{
    LayerGlobalBias* pLayer=new LayerGlobalBias(_iInSize);
	pLayer->weights() = _weight;

    return pLayer;
}
///////////////////////////////////////////////////////////////////////////////
void LayerGlobalBias::init()
{
    _weight.setZero(); // by default

    Layer::init();
}
///////////////////////////////////////////////////////////////////////////////
void LayerGlobalBias::forward(const MatrixFloat& mMatIn,MatrixFloat& mMatOut) const
{
    mMatOut = mMatIn.array() + _weight(0);
}
///////////////////////////////////////////////////////////////////////////////
void LayerGlobalBias::backpropagation(const MatrixFloat &mInput,const MatrixFloat &mGradientOut, MatrixFloat &mGradientIn)
{
    _gradientWeight = colWiseMean(mInput);
	
    mGradientIn = mGradientOut;
}
///////////////////////////////////////////////////////////////////////////////
float LayerGlobalBias::bias() const
{
    return _weight(0);
}
///////////////////////////////////////////////////////////////
bool LayerGlobalBias::has_weight() const
{
	return true;
}
///////////////////////////////////////////////////////////////
