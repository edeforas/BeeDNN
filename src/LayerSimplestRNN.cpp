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
    _h.setZero(1, _iUnits);

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
void LayerSimplestRNN::forward_frame(const MatrixFloat& mIn, MatrixFloat& mOut)
{
        _h = _h * _whh + mIn ;
        _h = tanh(_h);
		mOut=_h;
}
///////////////////////////////////////////////////////////////////////////////
void LayerSimplestRNN::backpropagation_frame(const MatrixFloat& mIn, const MatrixFloat& mGradientOut, MatrixFloat& mGradientIn)
{
    //Todo
}
/////////////////////////////////////////////////////////////////////////////////////////////