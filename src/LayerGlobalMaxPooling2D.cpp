/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#include "LayerGlobalMaxPooling2D.h"

///////////////////////////////////////////////////////////////////////////////
LayerGlobalMaxPooling2D::LayerGlobalMaxPooling2D(Index iInRows, Index iInCols, Index iInChannels) :
    Layer("GlobalMaxPooling2D")
{
	_iInRows = iInRows;
	_iInCols = iInCols;
	_iInChannels = iInChannels;
}
///////////////////////////////////////////////////////////////////////////////
LayerGlobalMaxPooling2D::~LayerGlobalMaxPooling2D()
{ }
///////////////////////////////////////////////////////////////////////////////
void LayerGlobalMaxPooling2D::get_params(Index& iInRows, Index& iInCols, Index & iInChannels) const
{
	iInRows = _iInRows;
	iInCols = _iInCols;
	iInChannels = _iInChannels;
}
///////////////////////////////////////////////////////////////////////////////
Layer* LayerGlobalMaxPooling2D::clone() const
{
    return new LayerGlobalMaxPooling2D(_iInRows, _iInCols, _iInChannels);
}
///////////////////////////////////////////////////////////////////////////////
void LayerGlobalMaxPooling2D::forward(const MatrixFloat& mIn,MatrixFloat& mOut)
{
	//not optimized yet
	mOut.resize( mIn.rows(),_iInChannels);
	if(_bTrainMode)
		_mMaxIndex.resizeLike(mOut); //index to selected input max data

	Index iBatchSize = mIn.rows();

	for(Index batch=0;batch<iBatchSize;batch++)
		for (Index channel = 0; channel < _iInChannels; channel++)
		{
			float fMax = -1.e38f;
			Index iPosMaxIn = -1;
			for (Index r = 0; r < _iInRows; r++)
			{
				for (Index c = 0; c < _iInCols; c++)
				{
					Index iPosIn = channel * _iInRows * _iInCols + r * _iInCols + c;
					float fSample = mIn(batch, iPosIn);
					if (fSample > fMax)
					{
						fMax = fSample;
						iPosMaxIn = iPosIn;
					}
				}
			}

			mOut(batch, channel) = fMax;
			if(_bTrainMode)
				_mMaxIndex(batch,channel)=(float)iPosMaxIn; // todo use Matrix<index>
		}
}
///////////////////////////////////////////////////////////////////////////////
void LayerGlobalMaxPooling2D::backpropagation(const MatrixFloat &mIn,const MatrixFloat &mGradientOut, MatrixFloat &mGradientIn)
{
    (void)mIn;

	if (_bFirstLayer)
		return;

	mGradientIn.setZero(mGradientOut.rows(), _iInChannels*_iInCols*_iInRows);

	for (Index batch = 0; batch < mGradientOut.rows(); batch++)
	{
		for (Index channel = 0; channel < _iInChannels; channel++)
		{
			mGradientIn(batch, (Index)_mMaxIndex(batch, channel)) = mGradientOut(batch,channel );
		}
	}
}
///////////////////////////////////////////////////////////////////////////////