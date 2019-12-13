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
LayerConvolution2D::LayerConvolution2D(int iInRows, int iInCols,int iInPlanes, int iKernelRows, int iKernelCols,int  iOutPlanes) :
    Layer(iInRows*iInCols*iInPlanes, 0 , "Convolution2D")
{
	_iInRows = iInRows;
	_iInCols = iInCols;
	_iInPlanes = iInPlanes;
	_iKernelRows = iKernelRows;
	_iKernelCols = iKernelCols;
	_iOutPlanes = iOutPlanes;
	
	_iBorderRows=iKernelRows>>1;
	_iBorderCols=iKernelCols>>1;

	_iOutRows=_iInRows-2* _iBorderRows;
	_iOutCols=_iInCols-2* _iBorderCols;

	_iOutSize = _iOutRows * _iOutCols*_iOutPlanes;

	LayerConvolution2D::init();
}
///////////////////////////////////////////////////////////////////////////////
LayerConvolution2D::~LayerConvolution2D()
{ }
///////////////////////////////////////////////////////////////////////////////
void LayerConvolution2D::get_params(int& iInRows, int& iInCols,int& iInPlanes, int& iKernelRows, int& iKernelCols,int& iOutPlanes)
{
	iInRows = _iInRows;
	iInCols = _iInCols;
	iInPlanes = _iInPlanes;
	iKernelRows = _iKernelRows;
	iKernelCols = _iKernelCols;
	iOutPlanes = _iOutPlanes;
}
///////////////////////////////////////////////////////////////////////////////
Layer* LayerConvolution2D::clone() const
{
    return new LayerConvolution2D(_iInRows, _iInCols,_iInPlanes,_iKernelRows,_iKernelCols,_iOutPlanes);
}
///////////////////////////////////////////////////////////////////////////////
void LayerConvolution2D::forward(const MatrixFloat& mIn,MatrixFloat& mOut)
{
	mOut.setZero(mIn.rows(), _iOutPlanes*_iOutRows*_iOutCols);
	for(int iSample=0; iSample<mIn.rows();iSample++)
	{
		for(int iPlane=0; iPlane<_iInPlanes;iPlane++)
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