/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#include "LayerUpSampling2D.h"

namespace beednn {

LayerUpSampling2D::LayerUpSampling2D(Index iInRows, Index iInCols, Index iInChannels, Index iRowFactor, Index iColFactor) :
	Layer("UpSampling2D")
{
	assert(iInRows > 0);
	assert(iInCols > 0);
	assert(iInChannels > 0);
	assert(iRowFactor > 0);
	assert(iColFactor > 0);

	_iInRows = iInRows;
	_iInCols = iInCols;
	_iInChannels = iInChannels;
	_iRowFactor = iRowFactor;
	_iColFactor = iColFactor;
}

LayerUpSampling2D::~LayerUpSampling2D()
{ }

Layer* LayerUpSampling2D::clone() const
{
	return new LayerUpSampling2D(_iInRows, _iInCols, _iInChannels, _iRowFactor, _iColFactor);
}

void LayerUpSampling2D::forward(const MatrixFloat& mIn, MatrixFloat& mOut)
{
	const Index iInPlaneSize = _iInRows * _iInCols;
	const Index iOutRows = _iInRows * _iRowFactor;
	const Index iOutCols = _iInCols * _iColFactor;
	const Index iOutPlaneSize = iOutRows * iOutCols;

	assert(mIn.cols() == iInPlaneSize * _iInChannels);
	mOut.resize(mIn.rows(), iOutPlaneSize * _iInChannels);

	for (Index sample = 0; sample < mIn.rows(); sample++)
	{
		for (Index channel = 0; channel < _iInChannels; channel++)
		{
			const float* pIn = mIn.row(sample).data() + channel * iInPlaneSize;
			float* pOut = mOut.row(sample).data() + channel * iOutPlaneSize;

			for (Index row = 0; row < _iInRows; row++)
			{
				for (Index col = 0; col < _iInCols; col++)
				{
					const float value = pIn[row * _iInCols + col];
					for (Index rowOffset = 0; rowOffset < _iRowFactor; rowOffset++)
					{
						for (Index colOffset = 0; colOffset < _iColFactor; colOffset++)
						{
							const Index outRow = row * _iRowFactor + rowOffset;
							const Index outCol = col * _iColFactor + colOffset;
							pOut[outRow * iOutCols + outCol] = value;
						}
					}
				}
			}
		}
	}
}

void LayerUpSampling2D::backpropagation(const MatrixFloat& mIn, const MatrixFloat& mGradientOut, MatrixFloat& mGradientIn)
{
	const Index iInPlaneSize = _iInRows * _iInCols;
	const Index iOutRows = _iInRows * _iRowFactor;
	const Index iOutCols = _iInCols * _iColFactor;
	const Index iOutPlaneSize = iOutRows * iOutCols;

	assert(mIn.cols() == iInPlaneSize * _iInChannels);
	assert(mGradientOut.rows() == mIn.rows());
	assert(mGradientOut.cols() == iOutPlaneSize * _iInChannels);

	if (_bFirstLayer)
		return;

	mGradientIn.setZero(mIn.rows(), iInPlaneSize * _iInChannels);

	for (Index sample = 0; sample < mIn.rows(); sample++)
	{
		for (Index channel = 0; channel < _iInChannels; channel++)
		{
			const float* pGradientOut = mGradientOut.row(sample).data() + channel * iOutPlaneSize;
			float* pGradientIn = mGradientIn.row(sample).data() + channel * iInPlaneSize;

			for (Index row = 0; row < _iInRows; row++)
			{
				for (Index col = 0; col < _iInCols; col++)
				{
					float gradient = 0.f;
					for (Index rowOffset = 0; rowOffset < _iRowFactor; rowOffset++)
					{
						for (Index colOffset = 0; colOffset < _iColFactor; colOffset++)
						{
							const Index outRow = row * _iRowFactor + rowOffset;
							const Index outCol = col * _iColFactor + colOffset;
							gradient += pGradientOut[outRow * iOutCols + outCol];
						}
					}
					pGradientIn[row * _iInCols + col] = gradient;
				}
			}
		}
	}
}
}
