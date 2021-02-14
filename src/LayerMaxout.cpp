/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#include "LayerMaxout.h"

///////////////////////////////////////////////////////////////////////////////
LayerMaxout::LayerMaxout(Index iInSize, Index iOutSize) :
    Layer("Maxout")
{
	_iInSize = iInSize;
	_iOutSize = iOutSize;
	_iReduction = iInSize / iOutSize;
}
///////////////////////////////////////////////////////////////////////////////
LayerMaxout::~LayerMaxout()
{ }
///////////////////////////////////////////////////////////////////////////////
Layer* LayerMaxout::clone() const
{
    return new LayerMaxout(_iInSize, _iOutSize);
}
///////////////////////////////////////////////////////////////////////////////
void LayerMaxout::predict(const MatrixFloat& mIn, MatrixFloat& mOut)
{
	Index iRows = mIn.rows();
	mOut.resize(iRows, _iOutSize);

	if (mIn.rows() == 0)
		return;

	if (_bTrainMode)
		_mMaxIndex.resizeLike(mOut); //index to selected input max data

	for (Index r = 0; r < iRows; r++)
	{
		for (Index c = 0; c < _iOutSize; c++)
		{
			Index iPosIn = _iReduction * c;
			float fMax = mIn(r, iPosIn);

			for (Index redux = 1; redux < _iReduction; redux++)
			{
				Index ipos = _iReduction * c + redux;
				float f = mIn(r, ipos);
				if (f < fMax)
				{
					fMax = f;
					iPosIn = ipos;
				}
			}

			mOut(r, c) = fMax;

			if (_bTrainMode)
				_mMaxIndex(r, c) = (float)iPosIn;
		}
	}
}
///////////////////////////////////////////////////////////////////////////////
void LayerMaxout::backpropagation(const MatrixFloat &mIn,const MatrixFloat &mGradientOut, MatrixFloat &mGradientIn)
{
    (void)mIn;

	if (_bFirstLayer)
		return;
	
	mGradientIn.setZero(mGradientOut.rows(), _iInSize);
	
	for (Index r = 0; r < mGradientOut.rows(); r++)
		for (Index c = 0; c < mGradientOut.cols(); c++)
			mGradientIn( r, (Index)(_mMaxIndex(r, c))) = mGradientOut(r, c);
} 
///////////////////////////////////////////////////////////////////////////////