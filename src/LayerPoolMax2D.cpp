/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#include "LayerPoolMax2D.h"

///////////////////////////////////////////////////////////////////////////////
LayerPoolMax2D::LayerPoolMax2D(int iInRows, int iInCols, int iInPlanes, int iRowFactor, int iColFactor) :
    Layer(iInRows*iInCols, iInPlanes*iInRows*iInCols/(iRowFactor*iColFactor), "PoolMax2D")
{
	_iInRows = iInRows;
	_iInCols = iInCols;
	_iInPlanes = iInPlanes;
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
void LayerPoolMax2D::get_params(int& iInRows, int& iInCols, int & iPlanes, int& iRowFactor, int& iColFactor)
{
	iInRows = _iInRows;
	iInCols = _iInCols;
	iPlanes = _iInPlanes;
	iRowFactor = _iRowFactor;
	iColFactor= _iColFactor;
}
///////////////////////////////////////////////////////////////////////////////
Layer* LayerPoolMax2D::clone() const
{
    return new LayerPoolMax2D(_iInRows, _iInCols, _iInPlanes, _iRowFactor, _iColFactor);
}
///////////////////////////////////////////////////////////////////////////////
void LayerPoolMax2D::forward(const MatrixFloat& mIn,MatrixFloat& mOut)
{
	mOut = mIn;
	
	mOut.resize(mIn.rows(), _iOutRows* _iOutCols*_iInPlanes);
	if(_bTrainMode)
		_mMaxIndex.resizeLike(mOut); //index to selected input max data

	//not optimized yet
	for (int plane = 0; plane < _iInPlanes; plane++)
	{
		for (int l = 0; l < mIn.rows(); l++)
		{
			auto lIn = mIn.row(l+plane*_iInRows);
			auto lOut = mOut.row(l+plane*_iOutRows);
			for (int r = 0; r < _iOutRows; r++)
			{
				for (int c = 0; c < _iOutCols; c++)
				{
					float fMax = -1.e38f;
					int iPosIn = -1;
					
					for (int ri = r * _iRowFactor; ri < r*_iRowFactor + _iRowFactor; ri++)
						for (int ci = c * _iColFactor; ci < c*_iColFactor + _iColFactor; ci++)
						{
							int iIndex = ri * _iInCols + ci; //flat index
							float fSample = lIn(iIndex);
							
							if (fSample > fMax)
							{
								fMax = fSample;
								iPosIn = iIndex;
							}
						}
					
					int iIndexOut = r * _iOutCols + c;
					lOut(iIndexOut) = fMax;
					if (_bTrainMode)
						_mMaxIndex.row(l+plane*_iOutRows)(iIndexOut) = (float)iPosIn;
				}
			}
		}
	}
}
///////////////////////////////////////////////////////////////////////////////
void LayerPoolMax2D::backpropagation(const MatrixFloat &mIn,const MatrixFloat &mGradientOut, MatrixFloat &mGradientIn)
{
    (void)mIn;

	if (_bFirstLayer)
		return;

	mGradientIn.setZero(mGradientOut.rows(),_iInRows* _iInCols*_iInPlanes);

	for (int plane = 0; plane < _iInPlanes; plane++)
	{
		for (int l = 0; l < mGradientOut.rows(); l++)
		{
			auto lOut = mGradientOut.row(l+plane* _iInRows);
			auto lIn = mGradientIn.row(l+plane*_iOutRows);

			for(int i=0;i< mGradientOut.cols();i++)
				lIn((int)_mMaxIndex(i)) = lOut(i);
		}
	}
}
///////////////////////////////////////////////////////////////////////////////