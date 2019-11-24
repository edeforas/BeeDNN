/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#include "LayerPoolMax2D.h"

///////////////////////////////////////////////////////////////////////////////
LayerPoolMax2D::LayerPoolMax2D(int iInRows, int iInCols, int iRowFactor, int iColFactor) :
    Layer(iInRows*iInCols, iInRows*iInCols/(iRowFactor*iColFactor), "PoolMax2D")
{
	_iInRows = iInRows;
	_iInCols = iInCols;
	_iRowFactor = iRowFactor;
	_iColFactor = iColFactor;
	_iOutRows = iInRows/iRowFactor;
	_iOutCols = iInCols/iColFactor;

    LayerPoolMax2D::init();
}
///////////////////////////////////////////////////////////////////////////////
LayerPoolMax2D::~LayerPoolMax2D()
{ }
///////////////////////////////////////////////////////////////////////////////
Layer* LayerPoolMax2D::clone() const
{
    return new LayerPoolMax2D(_iInRows, _iInCols, _iRowFactor, _iColFactor);
}
///////////////////////////////////////////////////////////////////////////////
void LayerPoolMax2D::forward(const MatrixFloat& mIn,MatrixFloat& mOut)
{
	mOut.resize(_iOutRows, _iOutCols);
	if(_bTrainMode)
		_mMaxOrig.resize(_iOutRows, _iOutCols); //index to selected input max data

	//not optimized yet
	for (int r = 0; r < _iOutRows; r++)
		for (int c = 0; c < _iOutCols; c++)
		{
			float fMax = -1.e38f;
			int iPosIn = -1;

			for(int ri= r*_iRowFactor; ri< r*_iRowFactor+ _iRowFactor;ri++)
				for (int ci = c * _iColFactor; ci < c*_iColFactor + _iColFactor; ci++)
					if (mIn(ri, ci) > fMax)
					{
						fMax = mIn(ri, ci);
						iPosIn = ri * _iInCols + ci; //flat index
					}

			mOut(r, c) = fMax;
			if(_bTrainMode)
				_mMaxOrig(r, c) = (float)iPosIn;
		}
}
///////////////////////////////////////////////////////////////////////////////
void LayerPoolMax2D::backpropagation(const MatrixFloat &mIn,const MatrixFloat &mGradientOut, MatrixFloat &mGradientIn)
{
    (void)mIn;

	_gradientWeight.setZero(_iInRows, _iInCols);
	mGradientIn.resize(_iInRows, _iInCols);



	// uses _mMaxOrig


	/*
	mGradientIn.setZero(mIn.rows(),mIn.cols()); //todo optimize everything
	for (int l = 0; l < mIn.rows(); l++)
	{
		for (int iOut = 0; iOut < mGradientOut.cols(); iOut++)
			mGradientIn(l, (int)_gradientWeight(l, iOut)) = mGradientOut(l, iOut);
	}
	*/
}
///////////////////////////////////////////////////////////////////////////////