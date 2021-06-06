/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#include "LayerPoolMax2D.h"

///////////////////////////////////////////////////////////////////////////////
LayerPoolMax2D::LayerPoolMax2D(Index iInRows, Index iInCols, Index iInChannels, Index iRowFactor, Index iColFactor) :
    Layer("PoolMax2D")
{
	_iInRows = iInRows;
	_iInCols = iInCols;
	_iInChannels = iInChannels;
	_iRowFactor = iRowFactor;
	_iColFactor = iColFactor;
	_iOutRows = iInRows/iRowFactor;
	_iOutCols = iInCols/iColFactor;
	_iInPlaneSize = _iInRows * _iInCols;
	_iOutPlaneSize = _iOutRows * _iOutCols;

    LayerPoolMax2D::init();
}
///////////////////////////////////////////////////////////////////////////////
LayerPoolMax2D::~LayerPoolMax2D()
{ }
///////////////////////////////////////////////////////////////////////////////
void LayerPoolMax2D::get_params(Index& iInRows, Index& iInCols, Index & iInChannels, Index& iRowFactor, Index& iColFactor) const
{
	iInRows = _iInRows;
	iInCols = _iInCols;
	iInChannels = _iInChannels;
	iRowFactor = _iRowFactor;
	iColFactor= _iColFactor;
}
///////////////////////////////////////////////////////////////////////////////
Layer* LayerPoolMax2D::clone() const
{
    return new LayerPoolMax2D(_iInRows, _iInCols, _iInChannels, _iRowFactor, _iColFactor);
}
///////////////////////////////////////////////////////////////////////////////
void LayerPoolMax2D::forward(const MatrixFloat& mIn,MatrixFloat& mOut)
{
	mOut.resize(mIn.rows(), _iOutPlaneSize*_iInChannels);
	if(_bTrainMode)
		_mMaxIndex.resizeLike(mOut); //index to selected input max data

	//not optimized yet
	for (Index channel = 0; channel < _iInChannels; channel++)
	{
		for (Index l = 0; l < mIn.rows(); l++)
		{
			const float* lIn = mIn.row(l).data()+ channel * _iInPlaneSize;
			float* lOut = mOut.row(l).data()+channel* _iOutPlaneSize;
			for (Index r = 0; r < _iOutRows; r++)
			{
				for (Index c = 0; c < _iOutCols; c++)
				{
					float fMax = -1.e38f;
					Index iPosIn = -1;
					
					for (Index ri = r * _iRowFactor; ri < r*_iRowFactor + _iRowFactor; ri++)
					{
						for (Index ci = c * _iColFactor; ci < c*_iColFactor + _iColFactor; ci++)
						{
							Index iIndex = ri * _iInCols + ci; //flat index in plane
							float fSample = lIn[iIndex];

							if (fSample > fMax)
							{
								fMax = fSample;
								iPosIn = iIndex;
							}
						}
					}

					Index iIndexOut = r * _iOutCols + c;
					lOut[iIndexOut] = fMax;
					if (_bTrainMode)
						_mMaxIndex(l, channel*_iOutPlaneSize +iIndexOut) = (float)iPosIn;
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

	mGradientIn.setZero(mGradientOut.rows(), _iInPlaneSize*_iInChannels);

	for (Index r = 0; r < mGradientOut.rows(); r++)
	{
		for (Index channel = 0; channel < _iInChannels; channel++)
		{
			const float* lOut = mGradientOut.row(r).data() +channel * _iOutPlaneSize;
			float* lIn = mGradientIn.row(r).data() +channel * _iInPlaneSize;

			for (Index i = 0; i < _iOutPlaneSize; i++)
			{
				lIn[(Index)_mMaxIndex(i)] = lOut[i];
			}
		}
	}
}
///////////////////////////////////////////////////////////////////////////////