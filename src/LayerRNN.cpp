/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#include "LayerRNN.h"

///////////////////////////////////////////////////////////////////////////////
LayerRNN::LayerRNN(int iSampleSize, int iUnits) :
    Layer("RNN"),
    _iSampleSize(iSampleSize),
    _iUnits(iUnits)
{
//    LayerRNN::init();
}
///////////////////////////////////////////////////////////////////////////////
LayerRNN::~LayerRNN()
{ }
///////////////////////////////////////////////////////////////////////////////
void LayerRNN::forward(const MatrixFloat& mIn, MatrixFloat& mOut)
{
    assert( (mIn.cols() % _iSampleSize)==0); // all samples are concatened horizontaly

    Index iNbSamples = mIn.rows();

    if (mIn.size() != _iSampleSize)
    {
        // not on-the-fly prediction, reset state on startup
        init();
    }

    MatrixFloat mSample;
    for (Index iS = 0; iS < mIn.cols(); iS+=_iSampleSize)
    {
      //  mSample = colView(mIn, iS , iS + _iSampleSize); //TODO
		forward_sample(mSample,mOut);
    }
}
///////////////////////////////////////////////////////////////////////////////
void LayerRNN::backpropagation(const MatrixFloat& mIn, const MatrixFloat& mGradientOut, MatrixFloat& mGradientIn)
{
    //Todo
}
/////////////////////////////////////////////////////////////////////////////////////////////
void LayerRNN::init()
{
    Layer::init();
}
/////////////////////////////////////////////////////////////////////////////////////////////