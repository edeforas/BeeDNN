/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#include "LayerCRelu.h"
namespace bee{


// CRelu as in : https://arxiv.org/pdf/1603.05201.pdf
// Warning: double the output size
// 
///////////////////////////////////////////////////////////////////////////////
LayerCRelu::LayerCRelu() :
    Layer("CRelu")
{	
    LayerCRelu::init();
}
///////////////////////////////////////////////////////////////////////////////
LayerCRelu::~LayerCRelu()
{ }
///////////////////////////////////////////////////////////////////////////////
Layer* LayerCRelu::clone() const
{
    return new LayerCRelu();
}
///////////////////////////////////////////////////////////////////////////////
void LayerCRelu::init()
{
    Layer::init();
}
///////////////////////////////////////////////////////////////////////////////
void LayerCRelu::forward(const MatrixFloat& mIn,MatrixFloat& mOut)
{
	Index iInSize=mIn.cols();
	mOut.setZero(mIn.rows(),iInSize*2);
	for (Index r = 0; r < mIn.rows(); r++)
		for (Index c = 0; c < iInSize; c++)
		{
			float f=mIn(r,c);
			if(f>=0.)
				mOut(r,c)=f;
			else
				mOut(r,c+iInSize)=-f;
		}			
}
///////////////////////////////////////////////////////////////////////////////
void LayerCRelu::backpropagation(const MatrixFloat &mIn,const MatrixFloat &mGradientOut, MatrixFloat &mGradientIn)
{
	if (_bFirstLayer)
		return;

	Index iInSize = mIn.cols();
	mGradientIn.setZero(mIn.rows(), iInSize);
	for (Index r = 0; r < mIn.rows(); r++)
		for (Index c = 0; c < iInSize; c++)
		{
			if (mIn(r, c) > 0.)
				mGradientIn(r, c) = mGradientOut(r, c);
			else
				mGradientIn(r, c) = -mGradientOut(r, c + iInSize);
		}
}
///////////////////////////////////////////////////////////////
}