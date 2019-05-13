/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#include "LayerPoolAveraging1D.h"

///////////////////////////////////////////////////////////////////////////////
LayerPoolAveraging1D::LayerPoolAveraging1D(int iInSize, int iOutSize) :
    Layer(iInSize , iOutSize, "PoolAveraging1D")
{
	assert(iOutSize>0);

    int iWindowSize=iInSize/iOutSize;
    float fInvWeight=1.f/iWindowSize;

    _weight.setZero(_iInSize,_iOutSize);

    for(int i=0;i<iOutSize;i++)
        for(int j=0;j<iWindowSize;j++)
            _weight(i*iWindowSize+j,i)=fInvWeight;
}
///////////////////////////////////////////////////////////////////////////////
LayerPoolAveraging1D::~LayerPoolAveraging1D()
{ }
///////////////////////////////////////////////////////////////////////////////
Layer* LayerPoolAveraging1D::clone() const
{
    return new LayerPoolAveraging1D(_iInSize,_iOutSize);
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
