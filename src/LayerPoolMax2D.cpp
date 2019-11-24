/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#include "LayerPoolMax2D.h"

///////////////////////////////////////////////////////////////////////////////
LayerPoolMax2D::LayerPoolMax2D(int iInRow, int iInCols, int iRowFactor, int iColFactor) :
    Layer(iInRow*iInCols, iInRow*iInCols/(iRowFactor*iColFactor), "PoolMax2D")
{
	_iInRow = iInRow;
	_iInCols = iInCols;
	_iRowFactor = iRowFactor;
	_iColFactor = iColFactor;

    LayerPoolMax2D::init();
}
///////////////////////////////////////////////////////////////////////////////
LayerPoolMax2D::~LayerPoolMax2D()
{ }
///////////////////////////////////////////////////////////////////////////////
Layer* LayerPoolMax2D::clone() const
{
    return new LayerPoolMax2D(_iInRow, _iInCols, _iRowFactor, _iColFactor);
}
///////////////////////////////////////////////////////////////////////////////
void LayerPoolMax2D::forward(const MatrixFloat& mIn,MatrixFloat& mOut)
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
void LayerPoolMax2D::backpropagation(const MatrixFloat &mIn,const MatrixFloat &mGradientOut, MatrixFloat &mGradientIn)
{
    (void)mIn;

	mGradientIn.setZero(mIn.rows(),mIn.cols()); //todo optimize everything
	for (int l = 0; l < mIn.rows(); l++)
	{
		for (int iOut = 0; iOut < mGradientOut.cols(); iOut++)
			mGradientIn(l, (int)_gradientWeight(l, iOut)) = mGradientOut(l, iOut);
	}
}
///////////////////////////////////////////////////////////////////////////////