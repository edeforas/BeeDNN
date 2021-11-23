/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#include "LayerSimplestRNN.h"

///////////////////////////////////////////////////////////////////////////////
LayerSimplestRNN::LayerSimplestRNN(int iSampleSize) :
    LayerRNN(iSampleSize, iSampleSize)
{
    LayerSimplestRNN::init();
}
///////////////////////////////////////////////////////////////////////////////
LayerSimplestRNN::~LayerSimplestRNN()
{ }
///////////////////////////////////////////////////////////////////////////////
void LayerSimplestRNN::init()
{
    _whh.setRandom(_iUnits, _iUnits); // Todo Xavier init ?

    LayerRNN::init();
}
///////////////////////////////////////////////////////////////////////////////
Layer* LayerSimplestRNN::clone() const
{
    LayerSimplestRNN* pLayer=new LayerSimplestRNN(_iSampleSize);
	pLayer->_whh = _whh;
    pLayer->_h = _h;

    return pLayer;
}
///////////////////////////////////////////////////////////////////////////////
void LayerSimplestRNN::forward_frame(const MatrixFloat& mInFrame, MatrixFloat& mOut)
{
    if (_h.rows() != mInFrame.rows())  // adapt to batch size
        _h.setZero(mInFrame.rows(), _iUnits);

    _h = tanh(_h * _whh + mInFrame);
	mOut=_h;
}
///////////////////////////////////////////////////////////////////////////////
void LayerSimplestRNN::backpropagation_frame(const MatrixFloat& mInFrame, const MatrixFloat& mGradientOut, MatrixFloat& mGradientIn)
{
    //grad(L/_Whh)=grad(L/h(t))*h(t-1)*(1-h(t)**2)
    //grad(L/h(t-1))=grad(L/h(t))*_Whh*(1-h(t)**2)
}
/////////////////////////////////////////////////////////////////////////////////////////////