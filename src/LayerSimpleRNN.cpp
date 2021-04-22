/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#include "LayerSimpleRNN.h"

///////////////////////////////////////////////////////////////////////////////
LayerSimpleRNN::LayerSimpleRNN(int iSampleSize, int iUnits) :
    Layer("SimpleRNN"),
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
    _whh.setRandom(_iUnits, _iUnits); // Todo Xavier init ?
    _wxh.setRandom(_iSampleSize, _iUnits); // Todo Xavier init ?
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
void LayerSimpleRNN::forward(const MatrixFloat& mIn, MatrixFloat& mOut)
{
    assert( (mIn.cols() % _iSampleSize)==0); // all samples are concatened horizontaly

    Index iNbSamples = mIn.rows();

    _h.resize(iNbSamples, _iUnits);
    if (mIn.size() != _iSampleSize)
    {
        // not on-the-fly prediction, reset state on startup
        _h.setZero();
    }

    Index iNbStep = mIn.cols() / _iSampleSize;

    MatrixFloat mX;
    for (Index i = 0; i < iNbStep; i++)
    {
        mX = colView(mIn, i * _iSampleSize, i * _iSampleSize + _iSampleSize);

        MatrixFloat a = _h * _whh;
        MatrixFloat b = mX * _wxh;
        MatrixFloat c = _bh;

        _h = _h * _whh + mX * _wxh;
        rowWiseAdd(_h, _bh);
        _h = tanh(_h);
    }

    mOut = _h;
}
///////////////////////////////////////////////////////////////////////////////
void LayerSimpleRNN::backpropagation(const MatrixFloat& mIn, const MatrixFloat& mGradientOut, MatrixFloat& mGradientIn)
{

}
/////////////////////////////////////////////////////////////////////////////////////////////