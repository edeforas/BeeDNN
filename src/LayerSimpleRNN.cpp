/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#include "LayerSimpleRNN.h"

///////////////////////////////////////////////////////////////////////////////
LayerSimpleRNN::LayerSimpleRNN(int iSampleSize, int iUnits) :
    LayerRNN("SimpleRNN"),
    _iSampleSize(iSampleSize),
    _iUnits(iUnits)
{
    LayerSimpleRNN::init();
}
///////////////////////////////////////////////////////////////////////////////
LayerSimpleRNN::~LayerSimpleRNN()
{ }
///////////////////////////////////////////////////////////////////////////////
void LayerSimpleRNN::init()
{
    _whh.setRandom(_iUnits, _iUnits);
    _wxh.setRandom(_iSampleSize, _iUnits);
    _bh.setZero(1, _iUnits);
    _h.setZero(1, _iUnits);

	Layer::init();
}
///////////////////////////////////////////////////////////////////////////////
Layer* LayerSimpleRNN::clone() const
{
    LayerSimpleRNN* pLayer=new LayerSimpleRNN(_iSampleSize,_iUnits);
	pLayer->_whh = _whh;
    pLayer->_wxh = _wxh;
    pLayer->_bh = _bh;
    pLayer->_h = _h;

    return pLayer;
}
///////////////////////////////////////////////////////////////////////////////
void LayerSimpleRNN::step(const MatrixFloat& mIn, MatrixFloat& mOut)
{
    assert(mIn.cols() == _iSampleSize);
    assert(mIn.rows() == 1);

    mOut = tanh(_whh*_h+_wxh*mIn+_bh);
}
/////////////////////////////////////////////////////////////////////////////////////////////