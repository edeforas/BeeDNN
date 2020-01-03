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
	
//	_iKernelSize = _iKernelRows * _iKernelCols*_iInChannels*_iOutChannels;

	_iBorderRows=iKernelRows>>1;
	_iBorderCols=iKernelCols>>1;

	_iOutRows=_iInRows-2* _iBorderRows;
	_iOutCols=_iInCols-2* _iBorderCols;

//	_iOutSize = _iOutRows * _iOutCols*iOutChannels;

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
	mOut = _weight * _im2col;
	mOut.resize(_iOutChannels, _iOutRows * _iOutCols); //use conservativeResize ? 
													   //todo reshape correctly
}
///////////////////////////////////////////////////////////////////////////////
void LayerConvolution2D::backpropagation(const MatrixFloat &mIn,const MatrixFloat &mGradientOut, MatrixFloat &mGradientIn)
{
	//from dense layer:

	_gradientWeight = (mIn.transpose())*mGradientOut*(1.f / mIn.rows());

	if (_bFirstLayer)
		return;

	mGradientIn = mGradientOut * (_weight.transpose());
}
///////////////////////////////////////////////////////////////////////////////
void LayerConvolution2D::init()
{
	_weight.resize(_iOutChannels, _iKernelRows * _iKernelCols*_iInChannels);
	setRandomUniform(_weight);

	_gradientWeight.resizeLike(_weight);
	_gradientWeight.setZero();

	_im2col.resize(_iKernelRows * _iKernelCols*_iInChannels, _iOutRows * _iOutCols* _iInCols);
}
///////////////////////////////////////////////////////////////////////////////
void LayerConvolution2D::im2col(const MatrixFloat & mIn)
{
	//todo
}
///////////////////////////////////////////////////////////////////////////////
