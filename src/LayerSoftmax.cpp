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
    mOut= mIn.array().exp();

    for(int i=0;i<mOut.rows();i++)
        mOut.row(i)/=mOut.row(i).sum();

 //   arraySub(mOut,mIn.maxCoeff()); // todo simplify and optimize
}
///////////////////////////////////////////////////////////////////////////////
// from https://medium.com/@14prakash/back-propagation-is-very-simple-who-made-it-complicated-97b794c97e5c
void LayerSoftmax::backpropagation(const MatrixFloat &mInput,const MatrixFloat &mDelta, MatrixFloat &mNewDelta)
{
    MatrixFloat S;
    mNewDelta.resize(mDelta.rows(),mDelta.cols());

	for (int r = 0; r < mInput.rows(); r++)
	{
	//	forward(mInput.row(r), S);
		S = mInput.row(r);
		float s = S.array().exp().sum();
		for (int c = 0; c < S.cols(); c++)
		{
			float expx = expf(S(c));

			mNewDelta(r, c) = mDelta(r, c)*(expx*(s-expx)) / (s*s);
		}
	}
}
///////////////////////////////////////////////////////////////////////////////
