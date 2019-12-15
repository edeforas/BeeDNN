/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#include "LayerPoolMax2D.h"

///////////////////////////////////////////////////////////////////////////////
LayerPoolMax2D::LayerPoolMax2D(int iRowFactor, int iColFactor) :
    Layer("PoolMax2D")
{
	_iRowFactor = iRowFactor;
	_iColFactor = iColFactor;
}
///////////////////////////////////////////////////////////////////////////////
LayerPoolMax2D::~LayerPoolMax2D()
{ }
///////////////////////////////////////////////////////////////////////////////
void LayerPoolMax2D::get_params(int& iRowFactor, int& iColFactor)
{
	iRowFactor = _iRowFactor;
	iColFactor= _iColFactor;
}
///////////////////////////////////////////////////////////////////////////////
void LayerPoolMax2D::init()
{
	_iOutputRows = _iInputRows / _iRowFactor;
	_iOutputCols = _iInputCols / _iColFactor;
	_iInPlaneSize = _iInputRows * _iInputCols;
	_iOutPlaneSize = _iOutputRows * _iOutputCols;
}
///////////////////////////////////////////////////////////////////////////////
Layer* LayerPoolMax2D::clone() const
{
    return new LayerPoolMax2D(_iRowFactor, _iColFactor);
}
///////////////////////////////////////////////////////////////////////////////
void LayerPoolMax2D::forward(const MatrixFloat& mIn,MatrixFloat& mOut)
{
	mOut = mIn;
	
	mOut.resize(mIn.rows(), _iOutPlaneSize*_iInputPlanes);
	if(_bTrainMode)
		_mMaxIndex.resizeLike(mOut); //index to selected input max data

	//not optimized yet
	for (int plane = 0; plane < _iInputPlanes; plane++)
	{
		for (int l = 0; l < mIn.rows(); l++)
		{
			const float* lIn = mIn.row(l).data()+ plane * _iInPlaneSize;
			float* lOut = mOut.row(l).data()+plane* _iOutPlaneSize;
			for (int r = 0; r < _iOutRows; r++)
			{
				for (int c = 0; c < _iOutCols; c++)
				{
					float fMax = -1.e38f;
					int iPosIn = -1;
					
					for (int ri = r * _iRowFactor; ri < r*_iRowFactor + _iRowFactor; ri++)
					{
						for (int ci = c * _iColFactor; ci < c*_iColFactor + _iColFactor; ci++)
						{
							int iIndex = ri * _iInputCols + ci; //flat index in plane
							float fSample = lIn[iIndex];

							if (fSample > fMax)
							{
								fMax = fSample;
								iPosIn = iIndex;
							}
						}
					}

					int iIndexOut = r * _iOutCols + c;
					lOut[iIndexOut] = fMax;
					if (_bTrainMode)
						_mMaxIndex(l,plane*_iOutPlaneSize +iIndexOut) = (float)iPosIn;
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

	mGradientIn.setZero(mGradientOut.rows(), _iInPlaneSize*_iInputPlanes);

	for (int l = 0; l < mGradientOut.rows(); l++)
	{
		for (int plane = 0; plane < _iInputPlanes; plane++)
		{
			const float* lOut = mGradientOut.row(l).data() +plane * _iOutPlaneSize;
			float* lIn = mGradientIn.row(l).data() +plane * _iInPlaneSize;

			for (int i = 0; i < _iOutPlaneSize; i++)
			{
				lIn[(int)_mMaxIndex(i)] = lOut[i];
			}
		}
	}
}
///////////////////////////////////////////////////////////////////////////////