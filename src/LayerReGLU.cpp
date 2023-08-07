/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

// ReGLU as in : https://kikaben.com/swiglu-2020/

#include "LayerReGLU.h"
#include "Activations.h"

///////////////////////////////////////////////////////////////////////////////
LayerReGLU::LayerReGLU() :
    Layer("ReGLU")
{
	_pActivation = get_activation("Relu");
    LayerReGLU::init();
}
///////////////////////////////////////////////////////////////////////////////
LayerReGLU::~LayerReGLU()
{
	delete _pActivation;
}
///////////////////////////////////////////////////////////////////////////////
Layer* LayerReGLU::clone() const
{
    return new LayerReGLU();
}
///////////////////////////////////////////////////////////////////////////////
void LayerReGLU::init()
{
    Layer::init();
}
///////////////////////////////////////////////////////////////////////////////
void LayerReGLU::forward(const MatrixFloat& mIn, MatrixFloat& mOut)
{
	assert(((mIn.cols() & 1) == 0) && "mIn must have an even size");

	Index iNbCols=mIn.cols();
	Index iNbColsHalf= iNbCols/2;

	mOut.resize(mIn.rows(), iNbColsHalf);
	for (int r = 0; r < mIn.rows(); r++)
		for (int c = 0; c < iNbColsHalf; c++)
			mOut(r, c) = _pActivation->apply(mIn(r, c))*mIn(r, c + iNbColsHalf);
}
///////////////////////////////////////////////////////////////////////////////
void LayerReGLU::backpropagation(const MatrixFloat &mIn,const MatrixFloat &mGradientOut, MatrixFloat &mGradientIn)
{
	if (_bFirstLayer)
		return;

	mGradientIn.resizeLike(mIn);
	Index iNbCols = mIn.cols();
	Index iNbColsHalf = iNbCols / 2;

	// 1st part with activation and 2nd part without activation
	for (int r = 0; r < mIn.rows(); r++)
		for (int c = 0; c < iNbColsHalf; c++)
		{
			mGradientIn(r, c) = mGradientOut(r, c) * mIn(r, c + iNbColsHalf) * _pActivation->derivation(mIn(r, c)); // (dL/dt)*g(y)*f'(x)
			mGradientIn(r, c + iNbColsHalf) = mGradientOut(r, c) * _pActivation->apply(mIn(r, c)); // (dL/dt)*f(x)*g'(y)
		}
}
///////////////////////////////////////////////////////////////