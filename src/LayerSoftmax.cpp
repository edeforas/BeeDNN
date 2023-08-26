/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#include "LayerSoftmax.h"

#include <cmath>
namespace beednn {

///////////////////////////////////////////////////////////////////////////////
LayerSoftmax::LayerSoftmax():
    Layer("Softmax")
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
void LayerSoftmax::forward(const MatrixFloat& mIn,MatrixFloat& mOut)
{
	MatrixFloat S;
	mOut=mIn;

	for (Index r = 0; r < mOut.rows(); r++)// todo simplify and optimize
	{
		S = mOut.row(r); 
		S.array() -= S.maxCoeff(); //remove max
		S = S.array().exp();
		mOut.row(r) =S/ S.sum();
	}
}
///////////////////////////////////////////////////////////////////////////////
// from https://medium.com/@14prakash/back-propagation-is-very-simple-who-made-it-complicated-97b794c97e5c
// read also https://deepnotes.io/softmax-crossentropy
void LayerSoftmax::backpropagation(const MatrixFloat &mIn,const MatrixFloat &mGradientOut, MatrixFloat &mGradientIn)
{
	if (_bFirstLayer)
		return;

    MatrixFloat S;
	mGradientIn=mGradientOut;

	for (Index r = 0; r < mIn.rows(); r++) // todo simplify and optimize
	{
		// todo compute from mOut
		S = mIn.row(r);
		S.array() -= S.maxCoeff(); //remove max
		S = S.array().exp();
		S/= S.sum();

		for (Index c = 0; c < S.cols(); c++)
		{
			float p = S(c);
			mGradientIn(r, c) *=  ( p * (1.f-p) );
		}
	}
}
///////////////////////////////////////////////////////////////////////////////
}