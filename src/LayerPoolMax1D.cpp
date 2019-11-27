/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#include "LayerPoolMax1D.h"

///////////////////////////////////////////////////////////////////////////////
LayerPoolMax1D::LayerPoolMax1D(int iInSize, int iOutSize) :
    Layer(iInSize , iOutSize, "PoolMax1D")
{
    LayerPoolMax1D::init();
}
///////////////////////////////////////////////////////////////////////////////
LayerPoolMax1D::~LayerPoolMax1D()
{ }
///////////////////////////////////////////////////////////////////////////////
Layer* LayerPoolMax1D::clone() const
{
    return new LayerPoolMax1D(_iInSize,_iOutSize);
}
///////////////////////////////////////////////////////////////////////////////
void LayerPoolMax1D::forward(const MatrixFloat& mIn,MatrixFloat& mOut)
{
	mOut.resize(mIn.rows(), _iOutSize);
	int iWindowSize=_iInSize/_iOutSize;
	
	if (_bTrainMode)
	{
		_gradientWeight.setZero(mIn.rows(), _iOutSize);
	}

	for (int l = 0; l < mIn.rows(); l++)
	{
		int iOut = 0;

		for (int iw = 0; iw < _iOutSize; iw++) //todo optimize everything
		{
			int iStart = iw * iWindowSize;
			int iEnd = iStart + iWindowSize;
			int iMaxPos = 0;
			float fMax = mIn(l, iStart);

			for(int i= iStart;i< iEnd;i++)
				if (fMax<mIn(l, i) )
				{
					fMax = mIn(l, i);
					iMaxPos = i;
				}

			mOut(l, iw) = fMax;

			if (_bTrainMode)
			{
				//update gradient
				_gradientWeight(l, iOut) = (float)iMaxPos;
			}
		
			iOut++;
		}
	}
}
///////////////////////////////////////////////////////////////////////////////
void LayerPoolMax1D::backpropagation(const MatrixFloat &mIn,const MatrixFloat &mGradientOut, MatrixFloat &mGradientIn)
{
    (void)mIn;

	if (_bFirstLayer)
		return;

	mGradientIn.setZero(mIn.rows(),mIn.cols()); //todo optimize everything
	for (int l = 0; l < mIn.rows(); l++)
	{
		for (int iOut = 0; iOut < mGradientOut.cols(); iOut++)
			mGradientIn(l, (int)_gradientWeight(l, iOut)) = mGradientOut(l, iOut);
	}
}
///////////////////////////////////////////////////////////////////////////////