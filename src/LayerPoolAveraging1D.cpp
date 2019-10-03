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
    LayerPoolAveraging1D::init();
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
void LayerPoolAveraging1D::init()
{
    assert(_iOutSize>0);

    int iWindowSize=_iInSize/_iOutSize;
    float fInvWeight=1.f/iWindowSize;

    _weight.setZero(_iInSize,_iOutSize);

    for(int i=0;i<_iOutSize;i++)
        for(int j=0;j<iWindowSize;j++)
            _weight(i*iWindowSize+j,i)=fInvWeight; //todo do not use a diagonal matrix for this computation
}
///////////////////////////////////////////////////////////////////////////////
void LayerPoolAveraging1D::forward(const MatrixFloat& mIn,MatrixFloat& mOut) const
{
    mOut = mIn * _weight; //todo do not use a diagonal matrix for this computation
}
///////////////////////////////////////////////////////////////////////////////
void LayerPoolAveraging1D::backpropagation(const MatrixFloat &mIn,const MatrixFloat &mGradientOut, MatrixFloat &mGradientIn)
{
    (void)mIn;

    mGradientIn = mGradientOut * (_weight.transpose()); //to do , do not update weight!!!!!!!!!!!!!!!!! FAIL
}
///////////////////////////////////////////////////////////////////////////////
bool LayerPoolAveraging1D::has_weight() const
{
	return false;
}
///////////////////////////////////////////////////////////////////////////////