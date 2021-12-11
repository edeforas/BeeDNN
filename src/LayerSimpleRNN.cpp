/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#include "LayerSimpleRNN.h"

///////////////////////////////////////////////////////////////////////////////
LayerSimpleRNN::LayerSimpleRNN(int iSampleSize, int iUnits) :
    LayerRNN(iSampleSize, iUnits)
{
    LayerSimpleRNN::init();
}
///////////////////////////////////////////////////////////////////////////////
LayerSimpleRNN::~LayerSimpleRNN()
{ }
///////////////////////////////////////////////////////////////////////////////
void LayerSimpleRNN::init()
{
    _whh.setRandom(_iUnits, _iUnits); // Todo Xavier init ?
    _wxh.setRandom(_iFrameSize, _iUnits); // Todo Xavier init ?
    _bh.setZero(1, _iUnits);
    _h.setZero(1, _iUnits);

    LayerRNN::init();
}
///////////////////////////////////////////////////////////////////////////////
Layer* LayerSimpleRNN::clone() const
{
    LayerSimpleRNN* pLayer=new LayerSimpleRNN(_iFrameSize,_iUnits);
	pLayer->_whh = _whh;
    pLayer->_wxh = _wxh;
    pLayer->_bh = _bh;
    pLayer->_h = _h;

    return pLayer;
}
///////////////////////////////////////////////////////////////////////////////
void LayerSimpleRNN::forward_frame(const MatrixFloat& mIn, MatrixFloat& mOut)
{
        _h = _h * _whh + mIn * _wxh;
        rowWiseAdd(_h, _bh);
        _h = tanh(_h);
		mOut=_h;
}
///////////////////////////////////////////////////////////////////////////////
void LayerSimpleRNN::backpropagation_frame(const MatrixFloat& mInFrame, const MatrixFloat& mH, const MatrixFloat& mHm1, const MatrixFloat& mGradientOut, MatrixFloat& mGradientIn)
{
    //Todo
}
/////////////////////////////////////////////////////////////////////////////////////////////