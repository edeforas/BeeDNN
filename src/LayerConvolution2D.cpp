/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

//for now, stride=1, no bias (add LayerBias just after this one, for now) , mode ='valid'

#include "LayerConvolution2D.h"

///////////////////////////////////////////////////////////////////////////////
LayerConvolution2D::LayerConvolution2D(int iInRows, int iInCols, int iInChannels, int iKernelRows, int iKernelCols, int iOutChannels) :
    Layer(iInRows*iInCols*iInChannels, 0 , "Convolution2D")
{
	_iInRows = iInRows;
	_iInCols = iInCols;
	_iInChannels = iInChannels;
	_iSamples = 0;// set in forward()
	_iKernelRows = iKernelRows;
	_iKernelCols = iKernelCols;
	_iOutChannels = iOutChannels;
	
	_iBorderRows=iKernelRows>>1;
	_iBorderCols=iKernelCols>>1;

	_iOutRows=_iInRows-2* _iBorderRows;
	_iOutCols=_iInCols-2* _iBorderCols;

	LayerConvolution2D::init();
}
///////////////////////////////////////////////////////////////////////////////
LayerConvolution2D::~LayerConvolution2D()
{ }
///////////////////////////////////////////////////////////////////////////////
void LayerConvolution2D::get_params(int& iInRows, int& iInCols,int& iInChannels, int& iKernelRows, int& iKernelCols,int& iOutChannels)
{
	iInRows = _iInRows;
	iInCols = _iInCols;
	iInChannels = _iInChannels;
	iKernelRows = _iKernelRows;
	iKernelCols = _iKernelCols;
	iOutChannels = _iOutChannels;
}
///////////////////////////////////////////////////////////////////////////////
Layer* LayerConvolution2D::clone() const
{
    return new LayerConvolution2D(_iInRows, _iInCols, _iInChannels,_iKernelRows,_iKernelCols,_iOutChannels);
}
///////////////////////////////////////////////////////////////////////////////
void LayerConvolution2D::forward(const MatrixFloat& mIn,MatrixFloat& mOut)
{
	im2col(mIn);
	mOut = _weight * _im2col;
	col2im(mOut);
}
///////////////////////////////////////////////////////////////////////////////
void LayerConvolution2D::backpropagation(const MatrixFloat &mIn,const MatrixFloat &mGradientOut, MatrixFloat &mGradientIn)
{
	(void)mIn;
	assert(mGradientOut.rows() == _iSamples);
	assert(mGradientOut.cols() == _iOutRows * _iOutCols*_iOutChannels);
	
	//from dense layer: todo?
	//todo
	_gradientWeight = (mIn.transpose())*mGradientOut*(1.f / mIn.rows());

	assert(_gradientWeight.rows() == _weight.rows());
	assert(_gradientWeight.cols() == _weight.cols());

	if (_bFirstLayer)
		return;

	//todo
	mGradientIn = mGradientOut * (_weight.transpose());

	assert(mGradientIn.rows() == _iSamples);
	assert(mGradientIn.cols() == _iInRows * _iInCols*_iInChannels);
}
///////////////////////////////////////////////////////////////////////////////
void LayerConvolution2D::init()
{
	_weight.resize(_iOutChannels, _iKernelRows * _iKernelCols * _iInChannels);
	setRandomUniform(_weight);

	_gradientWeight.resizeLike(_weight);
	_gradientWeight.setZero();
}
///////////////////////////////////////////////////////////////////////////////
void LayerConvolution2D::im2col(const MatrixFloat & mIn)
{
	//for now, no optimsations
	_iSamples = (int)mIn.rows();
	_im2col.resize(_iKernelRows * _iKernelCols*_iInChannels, _iOutRows * _iOutCols* _iSamples);

	for (int iSample = 0; iSample < _iSamples; iSample++)
	{
		for (int iInChannel = 0; iInChannel < _iInChannels; iInChannel++)
		{
			for (int iKRow = 0; iKRow < _iKernelRows; iKRow++)
			{
				for (int iKCol = 0; iKCol < _iKernelCols; iKCol++)
				{
					for (int iOutRow = 0; iOutRow < _iOutRows; iOutRow++)
					{
						for (int iOutCol = 0; iOutCol < _iOutCols; iOutCol++)
						{
							int iRowInPlane = iOutRow+iKRow;
							int iColInPlane = iOutCol+iKCol;

							float f = mIn(iSample, iInChannel*_iInRows*_iInCols + iRowInPlane*_iInCols+ iColInPlane);

							_im2col(iInChannel*_iKernelRows * _iKernelCols + iKRow * _iKernelCols + iKCol, iSample*_iOutCols*_iOutRows + iOutRow*_iOutCols + iOutCol) = f;
						}
					}
				}
			}
		}
	}
}
///////////////////////////////////////////////////////////////////////////////
void LayerConvolution2D::col2im(MatrixFloat & mOut)
{
	mOut.resize(_iSamples * _iOutChannels, _iOutRows * _iOutCols );
	_col2im = mOut; //for now use a copy, todo optimize?

	for (int iOutChannel = 0; iOutChannel < _iOutChannels; iOutChannel++)
	{
		for (int iSample = 0; iSample < _iSamples; iSample++)
		{
			int iOrigRow = iSample+ iOutChannel* _iSamples;
			int iDestRow = iOutChannel+ iSample* _iOutChannels;

			mOut.row(iDestRow) = _col2im.row(iOrigRow);
		}
	}
	mOut.resize(_iSamples, _iOutRows * _iOutCols * _iOutChannels);
}
///////////////////////////////////////////////////////////////////////////////
