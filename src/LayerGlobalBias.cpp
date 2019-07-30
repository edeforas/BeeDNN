/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#include "LayerGlobalBias.h"

///////////////////////////////////////////////////////////////////////////////
LayerGlobalBias::LayerGlobalBias() :
    Layer(0 , 0, "GlobalGain")
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
    LayerGlobalBias* pLayer=new LayerGlobalBias();
	pLayer->weights() = _weight;

    return pLayer;
}
///////////////////////////////////////////////////////////////////////////////
void LayerGlobalBias::init()
{
    _weight.setZero(); //init to one by default

    Layer::init();
}
///////////////////////////////////////////////////////////////////////////////
void LayerGlobalBias::forward(const MatrixFloat& mMatIn,MatrixFloat& mMatOut) const
{
//    mMatOut = mMatIn + _weight(0);   //TODO WIP
}
///////////////////////////////////////////////////////////////////////////////
void LayerGlobalBias::backpropagation(const MatrixFloat &mInput,const MatrixFloat &mDelta, MatrixFloat &mNewDelta)
{
    mNewDelta = mDelta;// * _weight(0);
    _mDeltaWeight = mInput*(mDelta.transpose()); //TODO WIP
}
///////////////////////////////////////////////////////////////////////////////
float LayerGlobalBias::gain() const
{
    return _weight(0);
}
///////////////////////////////////////////////////////////////////////////////
MatrixFloat& LayerGlobalBias::weights()
{
    return _weight;
}
///////////////////////////////////////////////////////////////////////////////
MatrixFloat& LayerGlobalBias::gradient_weights()
{
    return _mDeltaWeight;
}
///////////////////////////////////////////////////////////////////////////////
bool LayerGlobalBias::has_weight()
{
    return true;
}
///////////////////////////////////////////////////////////////
