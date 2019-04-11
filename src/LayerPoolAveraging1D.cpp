/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#include "LayerPoolAveraging1D.h"

///////////////////////////////////////////////////////////////////////////////
LayerPoolAveraging1D::LayerPoolAveraging1D(int iInSize, int iWindowSize) :
    Layer(iInSize , iInSize/iWindowSize, "PoolAveraging1D")
{
    _iWindowSize=iWindowSize;
    _weight.resize(_iInSize,_iOutSize);
    _weight.setConstant(1.f/iWindowSize);
}
///////////////////////////////////////////////////////////////////////////////
LayerPoolAveraging1D::~LayerPoolAveraging1D()
{ }
///////////////////////////////////////////////////////////////////////////////
Layer* LayerPoolAveraging1D::clone() const
{
    return new LayerPoolAveraging1D(_iInSize,_iWindowSize);
}
///////////////////////////////////////////////////////////////////////////////
void LayerPoolAveraging1D::forward(const MatrixFloat& mMatIn,MatrixFloat& mMatOut) const
{
    mMatOut = mMatIn * _weight;
}
///////////////////////////////////////////////////////////////////////////////
void LayerPoolAveraging1D::backpropagation(const MatrixFloat &mInput,const MatrixFloat &mDelta, MatrixFloat &mInputDelta)
{
    (void)mInput;
    mInputDelta = mDelta * (_weight.transpose());
}
///////////////////////////////////////////////////////////////////////////////
int LayerPoolAveraging1D::window_size() const
{
    return _iWindowSize;
}
///////////////////////////////////////////////////////////////////////////////
