/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#include "LayerAveragePooling2D.h"

///////////////////////////////////////////////////////////////////////////////
LayerAveragePooling2D::LayerAveragePooling2D(Index iInRows, Index iInCols, Index iInChannels, Index iRowFactor, Index iColFactor) :
    Layer("PAveragePooling2D")
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
	_fInvKernelSize = 1.f / (float)(_iRowFactor * _iColFactor);
}
///////////////////////////////////////////////////////////////////////////////
LayerAveragePooling2D::~LayerAveragePooling2D()
{ }
///////////////////////////////////////////////////////////////////////////////
void LayerAveragePooling2D::get_params(Index& iInRows, Index& iInCols, Index & iInChannels, Index& iRowFactor, Index& iColFactor) const
{
	iInRows = _iInRows;
	iInCols = _iInCols;
	iInChannels = _iInChannels;
	iRowFactor = _iRowFactor;
	iColFactor= _iColFactor;
}
///////////////////////////////////////////////////////////////////////////////
Layer* LayerAveragePooling2D::clone() const
{
    return new LayerAveragePooling2D(_iInRows, _iInCols, _iInChannels, _iRowFactor, _iColFactor);
}
///////////////////////////////////////////////////////////////////////////////
void LayerAveragePooling2D::forward(const MatrixFloat& mIn,MatrixFloat& mOut)
{
	mOut.resize(mIn.rows(), _iOutPlaneSize*_iInChannels);

	//not optimized yet
	for (Index batch = 0; batch < mIn.rows(); batch++)
	{
		for (Index channel = 0; channel < _iInChannels; channel++)
		{
			const float* lIn = mIn.row(batch).data()+ channel * _iInPlaneSize;
			float* lOut = mOut.row(batch).data()+channel* _iOutPlaneSize;
			for (Index r = 0; r < _iOutRows; r++)
			{
				for (Index c = 0; c < _iOutCols; c++)
				{
					float fSum= 0.f;
					
					for (Index ri = r * _iRowFactor; ri < r*_iRowFactor + _iRowFactor; ri++)
					{
						for (Index ci = c * _iColFactor; ci < c*_iColFactor + _iColFactor; ci++)
						{
							Index iIndex = ri * _iInCols + ci; //flat index in plane
							fSum += lIn[iIndex];
						}
					}

					lOut[r * _iOutCols + c] = fSum*_fInvKernelSize;
				}
			}
		}
	}
}
///////////////////////////////////////////////////////////////////////////////
void LayerAveragePooling2D::backpropagation(const MatrixFloat &mIn,const MatrixFloat &mGradientOut, MatrixFloat &mGradientIn)
{
    (void)mIn;

	if (_bFirstLayer)
		return;

	mGradientIn.setZero(mGradientOut.rows(), _iInPlaneSize * _iInChannels);
	/*
	//not optimized yet
	for (Index channel = 0; channel < _iInChannels; channel++)
	{
		for (Index l = 0; l < _iInRows; l++)
		{
			const float* lIn = mIn.row(l).data() + channel * _iInPlaneSize;
			float* lOut = mOut.row(l).data() + channel * _iOutPlaneSize;
			for (Index r = 0; r < _iOutRows; r++)
			{
				for (Index c = 0; c < _iOutCols; c++)
				{
					float fSum = -0.f;

					for (Index ri = r * _iRowFactor; ri < r * _iRowFactor + _iRowFactor; ri++)
					{
						for (Index ci = c * _iColFactor; ci < c * _iColFactor + _iColFactor; ci++)
						{
							Index iIndex = ri * _iInCols + ci; //flat index in plane
							fSum += lIn[iIndex];
						}
					}

					lOut[r * _iOutCols + c] = fSum * fInvKernelSize;
				}
			}
		}
	}




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
*/
}
///////////////////////////////////////////////////////////////////////////////