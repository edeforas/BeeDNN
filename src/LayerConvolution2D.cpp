/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

//for now, stride=1, no bias, mode ='valid'

#include "LayerConvolution2D.h"

///////////////////////////////////////////////////////////////////////////////
LayerConvolution2D::LayerConvolution2D(int iInRows, int iInCols, int iInChannels, int iKernelRows, int iKernelCols, int iOutChannels) :
    Layer(iInRows*iInCols*iInChannels, 0 , "Convolution2D")
{
	_iInRows = iInRows;
	_iInCols = iInCols;
	_iInChannels = iInChannels;
	_iKernelRows = iKernelRows;
	_iKernelCols = iKernelCols;
	_iOutChannels = iOutChannels;
	
	_iKernelSize = _iKernelRows * _iKernelCols*_iInChannels*_iOutChannels;

	_iBorderRows=iKernelRows>>1;
	_iBorderCols=iKernelCols>>1;

	_iOutRows=_iInRows-2* _iBorderRows;
	_iOutCols=_iInCols-2* _iBorderCols;

	_iOutSize = _iOutRows * _iOutCols*iOutChannels;

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
	mOut.setZero(mIn.rows(), _iOutChannels*_iOutRows*_iOutCols);

	for (int iSample = 0; iSample < mIn.rows(); iSample++)
	{
		convolution2d(mIn.row(iSample).data(), _weight.data(), mOut.row(iSample).data());
	}
}
///////////////////////////////////////////////////////////////////////////////
void LayerConvolution2D::backpropagation(const MatrixFloat &mIn,const MatrixFloat &mGradientOut, MatrixFloat &mGradientIn)
{
    (void)mIn;
	(void)mGradientOut;
	(void)mGradientIn;

	mGradientIn = mIn;

	if (_bFirstLayer)
		return;

	//todo

}
///////////////////////////////////////////////////////////////////////////////
void LayerConvolution2D::init()
{
	_weight.resize(_iKernelSize, 1);
	_weight.setRandom();

	_gradientWeight.setZero(_iKernelSize, 1);
}
///////////////////////////////////////////////////////////////////////////////
void LayerConvolution2D::im2col(const MatrixFloat & mIn)
{
}
///////////////////////////////////////////////////////////////////////////////
void LayerConvolution2D::col2im(MatrixFloat & mIn)
{
}
///////////////////////////////////////////////////////////////////////////////
// no bias, mode 'valid', no strides
// for now, uses im2col algorithm
void LayerConvolution2D::convolution2d(const float* fIn,const float *fKernel, float* fOut)
{
	//int iHalfKernelRow = iKernelRows >> 1;
	//int iHalfKernelCols = iKernelCols >> 1;

	int iOutRow = _iInRows - _iKernelRows + 1;
	int iOutCols = _iInCols - _iKernelCols + 1;
	int iOutPlaneSize = iOutRow * iOutCols;

	for (int iOutPlane = 0; iOutPlane < _iOutChannels; iOutPlane++)
	{
		// set output accumulator plane to zero	
		float* fOutPlane = fOut + iOutPlane * iOutPlaneSize;
		for (int i = 0; i < iOutPlaneSize; i++)
			fOutPlane[i] = 0.f;

		//now convol and accumulate every inputplane
		for (int iInPlane = 0; iInPlane < _iInChannels; iInPlane++)
		{



		}
	}


	(void)fKernel;
}
///////////////////////////////////////////////////////////////////////////////