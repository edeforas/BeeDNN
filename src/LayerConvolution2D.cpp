/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

//for now, stride=1, no bias, mode ='valid'

#include "LayerConvolution2D.h"

///////////////////////////////////////////////////////////////////////////////
void LayerConvolution2D::conv_and_add(const MatrixFloat& mImage,const MatrixFloat& mKernel, MatrixFloat&mOut)
{
	//todo
}
///////////////////////////////////////////////////////////////////////////////
LayerConvolution2D::LayerConvolution2D(int iKernelRows, int iKernelCols,int  iOutPlanes) :
    Layer( "Convolution2D")
{
	_iKernelRows = iKernelRows;
	_iKernelCols = iKernelCols;
	_iOutputPlanes = iOutPlanes;
	
	_iBorderRows=iKernelRows>>1;
	_iBorderCols=iKernelCols>>1;

	_iOutputRows=_iInputRows-2* _iBorderRows;
	_iOutputCols=_iInputCols-2* _iBorderCols;

	_iOutputSize = _iOutputRows * _iOutputCols*_iOutputPlanes;

	LayerConvolution2D::init();
}
///////////////////////////////////////////////////////////////////////////////
LayerConvolution2D::~LayerConvolution2D()
{ }
///////////////////////////////////////////////////////////////////////////////
void LayerConvolution2D::get_params(int& iKernelRows, int& iKernelCols,int& iOutPlanes)
{
	iKernelRows = _iKernelRows;
	iKernelCols = _iKernelCols;
	iOutPlanes = _iOutputPlanes;
}
///////////////////////////////////////////////////////////////////////////////
Layer* LayerConvolution2D::clone() const
{
    return new LayerConvolution2D(_iKernelRows,_iKernelCols,_iOutputPlanes);
}
///////////////////////////////////////////////////////////////////////////////
void LayerConvolution2D::forward(const MatrixFloat& mIn,MatrixFloat& mOut)
{
	mOut.setZero(mIn.rows(), _iOutputSize);
	for(int iSample=0; iSample<mIn.rows();iSample++)
	{
		for(int iPlane=0; iPlane<_iInputPlanes;iPlane++)
		{
			//Matrix mImage()
			
			//conv_and_add()
		
		}
	}
	

}
///////////////////////////////////////////////////////////////////////////////
void LayerConvolution2D::backpropagation(const MatrixFloat &mIn,const MatrixFloat &mGradientOut, MatrixFloat &mGradientIn)
{
    (void)mIn;

	if (_bFirstLayer)
		return;

	//todo

}
///////////////////////////////////////////////////////////////////////////////