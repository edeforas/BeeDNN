/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#include "LayerGlobalGain.h"

#include <cmath> // for sqrt

///////////////////////////////////////////////////////////////////////////////
LayerGlobalGain::LayerGlobalGain(int iInSize, float fGlobalGain) :
    Layer(iInSize , iInSize, "GlobalGain")
{
    _bLearnable=(fGlobalGain==0.f); //for now

    _weight.resize(1,1);
    _weight(0)=fGlobalGain;

    LayerGlobalGain::init();
}
///////////////////////////////////////////////////////////////////////////////
LayerGlobalGain::~LayerGlobalGain()
{ }
///////////////////////////////////////////////////////////////////////////////
Layer* LayerGlobalGain::clone() const
{
    LayerGlobalGain* pLayer=new LayerGlobalGain(_iInSize,_weight(0));
    return pLayer;
}
///////////////////////////////////////////////////////////////////////////////
void LayerGlobalGain::init()
{
    if(_bLearnable)
    {
        //Xavier uniform initialization
        float a =sqrtf(6.f/(_iInSize+_iOutSize));
        _weight.setRandom();
        _weight*=a;
    }

    Layer::init();
}
///////////////////////////////////////////////////////////////////////////////
void LayerGlobalGain::forward(const MatrixFloat& mMatIn,MatrixFloat& mMatOut) const
{
    mMatOut = mMatIn * _weight(0);
}
///////////////////////////////////////////////////////////////////////////////
void LayerGlobalGain::backpropagation(const MatrixFloat &mInput,const MatrixFloat &mDelta, MatrixFloat &mNewDelta)
{
    mNewDelta = mDelta * _weight(0);
    _mDeltaWeight = mInput*(mDelta.transpose());
}
///////////////////////////////////////////////////////////////////////////////
float LayerGlobalGain::gain() const
{
    return _weight(0);
}
///////////////////////////////////////////////////////////////////////////////
bool LayerGlobalGain::is_learned() const
{
    return _bLearnable;
}
///////////////////////////////////////////////////////////////////////////////
MatrixFloat& LayerGlobalGain::weights()
{
    return _weight;
}
///////////////////////////////////////////////////////////////////////////////
MatrixFloat& LayerGlobalGain::gradient_weights()
{
    return _mDeltaWeight;
}
///////////////////////////////////////////////////////////////////////////////
bool LayerGlobalGain::has_weight()
{
    return true;
}
///////////////////////////////////////////////////////////////
