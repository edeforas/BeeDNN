/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

// ZeroPadding2S as in : https://www.tensorflow.org/api_docs/python/tf/keras/layers/ZeroPadding2D 
// and
// https://deeplizard.com/learn/video/qSTv_m-KFk0

#include "LayerZeroPadding2D.h"
namespace beednn {

///////////////////////////////////////////////////////////////////////////////
LayerZeroPadding2D::LayerZeroPadding2D(Index iInRows, Index iInCols, Index iInChannels, Index iBorder) :
    Layer("ZeroPadding2D")
{
	_iInRows = iInRows;
	_iInCols = iInCols;
	_iInChannels = iInChannels;
	_iBorder = iBorder;
}
///////////////////////////////////////////////////////////////////////////////
LayerZeroPadding2D::~LayerZeroPadding2D()
{ }
///////////////////////////////////////////////////////////////////////////////
void LayerZeroPadding2D::get_params(Index& iInRows, Index& iInCols, Index & iInChannels, Index& iBorder) const
{
	iInRows = _iInRows;
	iInCols = _iInCols;
	iInChannels = _iInChannels;
	iBorder = _iBorder;
}
///////////////////////////////////////////////////////////////////////////////
Layer* LayerZeroPadding2D::clone() const
{
    return new LayerZeroPadding2D(_iInRows, _iInCols, _iInChannels, _iBorder);
}
///////////////////////////////////////////////////////////////////////////////
void LayerZeroPadding2D::forward(const MatrixFloat& mIn,MatrixFloat& mOut)
{
	Index iOutCol=_iInCols+2*_iBorder;
	Index iOutPlaneSize=(_iInRows+2*_iBorder)*iOutCol;
	Index iInPlaneSize=_iInRows*_iInCols;
	mOut.setZero(mIn.rows(), iOutPlaneSize*_iInChannels);

	//not optimized yet
	for (Index sample = 0; sample < mIn.rows(); sample++)
	{
		for (Index channel = 0; channel < _iInChannels; channel++)
		{
			const float* lIn = mIn.row(sample).data()+ channel * iInPlaneSize;
			float* lOut = mOut.row(sample).data()+channel* iOutPlaneSize;
			for (Index r = 0; r < _iInRows; r++)
			{
				for (Index c = 0; c < _iInCols; c++)
				{
					lOut[(r+_iBorder)*iOutCol+c+_iBorder] = lIn[r*_iInCols+c];
				}
			}
		}
	}
}
///////////////////////////////////////////////////////////////////////////////
void LayerZeroPadding2D::backpropagation(const MatrixFloat &mIn,const MatrixFloat &mGradientOut, MatrixFloat &mGradientIn)
{
    (void)mIn;

	if (_bFirstLayer)
		return;

	Index iOutCol=_iInCols+2*_iBorder;
	Index iOutPlaneSize=(_iInRows+2*_iBorder)*iOutCol;
	Index iInPlaneSize=_iInRows*_iInCols;

	mGradientIn.resize(mGradientOut.rows(), iInPlaneSize*_iInChannels);

	//not optimized yet
	for (Index sample = 0; sample < mGradientOut.rows(); sample++)
	{
		for (Index channel = 0; channel < _iInChannels; channel++)
		{
			float* lIn = mGradientIn.row(sample).data()+ channel * iInPlaneSize;
			const float* lOut = mGradientOut.row(sample).data()+channel* iOutPlaneSize;
			for (Index r = 0; r < _iInRows; r++)
			{
				for (Index c = 0; c < _iInCols; c++)
				{
					lIn[r*_iInCols+c]=lOut[(r+_iBorder)*iOutCol+c+_iBorder] ; 
				}
			}
		}
	}
}
///////////////////////////////////////////////////////////////////////////////
}