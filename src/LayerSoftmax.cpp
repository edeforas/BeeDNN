/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#include "LayerSoftmax.h"

#include "Activation.h"

///////////////////////////////////////////////////////////////////////////////
LayerSoftmax::LayerSoftmax():
    Layer(0,0,"Softmax")
{ }
///////////////////////////////////////////////////////////////////////////////
LayerSoftmax::~LayerSoftmax()
{ }
///////////////////////////////////////////////////////////////////////////////
Layer* LayerSoftmax::clone() const
{
    return new LayerSoftmax();
}
///////////////////////////////////////////////////////////////////////////////
void LayerSoftmax::forward(const MatrixFloat& mIn,MatrixFloat& mOut) const
{
	MatrixFloat S;
	mOut.resizeLike(mIn);

	for (int r = 0; r < mOut.rows(); r++)// todo simplify and optimize
	{
		S = mIn.row(r); 
		S.array() -= S.maxCoeff(); //remove max
		S = S.array().exp();
		mOut.row(r)=  S / S.sum();
	}
}
///////////////////////////////////////////////////////////////////////////////
// from https://medium.com/@14prakash/back-propagation-is-very-simple-who-made-it-complicated-97b794c97e5c
void LayerSoftmax::backpropagation(const MatrixFloat &mInput,const MatrixFloat &mGradientOut, MatrixFloat &mGradientIn)
{
    MatrixFloat S;
	mGradientIn.resizeLike(mGradientOut);

	for (int r = 0; r < mInput.rows(); r++) // todo simplify and optimize
	{
		S = mInput.row(r);
		S.array() -= S.maxCoeff(); //remove max

		float s = S.array().exp().sum();
		for (int c = 0; c < S.cols(); c++)
		{
			float expx = expf(S(c));
			mGradientIn(r, c) = mGradientOut(r, c)*(expx*(s-expx)) / (s*s);
		}
	}
}
///////////////////////////////////////////////////////////////////////////////
