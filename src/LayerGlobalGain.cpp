/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#include "LayerGlobalGain.h"

#include <cmath> // for sqrt
#include "Optimizer.h"

///////////////////////////////////////////////////////////////////////////////
LayerGlobalGain::LayerGlobalGain(int iInSize, float fGlobalGain) :
    Layer(iInSize , iInSize, "GlobalGain")//,
  //  _fGlobalGain(fGlobalGain)
{
    _bLearnable=(fGlobalGain==0.f); //temp code

    _globalGain.resize(1,1);
    _globalGain(0)=fGlobalGain;

    init();
}
///////////////////////////////////////////////////////////////////////////////
LayerGlobalGain::~LayerGlobalGain()
{ }
///////////////////////////////////////////////////////////////////////////////
Layer* LayerGlobalGain::clone() const
{
    LayerGlobalGain* pLayer=new LayerGlobalGain(_iInSize,_globalGain(0));
    return pLayer;
}
///////////////////////////////////////////////////////////////////////////////
void LayerGlobalGain::init()
{
    if(_bLearnable)
    {
        //Xavier uniform initialization
        float a =sqrtf(6.f/(_iInSize+_iOutSize));
        _globalGain.setRandom();
        _globalGain*=a;
    }

    Layer::init();
}
///////////////////////////////////////////////////////////////////////////////
void LayerGlobalGain::forward(const MatrixFloat& mMatIn,MatrixFloat& mMatOut) const
{
    mMatOut = mMatIn * _globalGain(0);
}
///////////////////////////////////////////////////////////////////////////////
void LayerGlobalGain::backpropagation(const MatrixFloat &mInput,const MatrixFloat &mDelta, MatrixFloat &mNewDelta)
{
    mNewDelta = mDelta * _globalGain(0);
    _mDx = mInput*(mDelta.transpose());
/*
    if(_bLearnable)
        pOptim->optimize(_globalGain, _mDx);
*/
}
///////////////////////////////////////////////////////////////////////////////
float LayerGlobalGain::gain() const
{
    return _globalGain(0);
}
///////////////////////////////////////////////////////////////////////////////
bool LayerGlobalGain::is_learned() const
{
    return _bLearnable;
}
///////////////////////////////////////////////////////////////////////////////
