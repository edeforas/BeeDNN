/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

// Bilinear as in : https://kikaben.com/swiglu-2020/

#include "LayerBilinear.h"
#include "Activations.h"

///////////////////////////////////////////////////////////////////////////////
LayerBilinear::LayerBilinear() :
    Layer("Bilinear")
{
	LayerBilinear::init();
}
///////////////////////////////////////////////////////////////////////////////
LayerBilinear::~LayerBilinear()
{ }
///////////////////////////////////////////////////////////////////////////////
Layer* LayerBilinear::clone() const
{
    return new LayerBilinear();
}
///////////////////////////////////////////////////////////////////////////////
void LayerBilinear::init()
{
    Layer::init();
}
///////////////////////////////////////////////////////////////////////////////
void LayerBilinear::forward(const MatrixFloat& mIn, MatrixFloat& mOut)
{
	assert( (mIn.cols() % 1) == 0); // mIn must have an even size

	Index iNbCols=mIn.cols();
	Index iNbColsHalf= iNbCols/2;

	mOut.resize(mIn.rows(), iNbColsHalf);
	for (int r = 0; r < mIn.rows(); r++)
		for (int c = 0; c < iNbColsHalf; c++)
			mOut(r, c) = mIn(r, c)*mIn(r, c + iNbColsHalf);
}
///////////////////////////////////////////////////////////////////////////////
void LayerBilinear::backpropagation(const MatrixFloat &mIn,const MatrixFloat &mGradientOut, MatrixFloat &mGradientIn)
{
	if (_bFirstLayer)
		return;
	
	mGradientIn.resizeLike(mIn);

	Index iNbCols = mIn.cols();
	Index iNbColsHalf = iNbCols / 2;

	for (int r = 0; r < mIn.rows(); r++)
		for (int c = 0; c < iNbColsHalf; c++)
		{
			mGradientIn(r, c) = mGradientOut(r, c)*mIn(r, c + iNbColsHalf); // (dL/dt)*g(y)*f'(x)
			mGradientIn(r, c + iNbColsHalf) = mGradientOut(r, c) * mIn(r, c); // (dL/dt)*f(x)*g'(y)
		}
}
///////////////////////////////////////////////////////////////