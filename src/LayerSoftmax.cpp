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
    mOut.resize(mIn.rows(),mIn.cols());

//    for(int i=0;i<mOut.size();i++)
//        mOut(i)=_pActivation->apply(mIn(i));

//todo
}
///////////////////////////////////////////////////////////////////////////////
void LayerSoftmax::backpropagation(const MatrixFloat &mInput,const MatrixFloat &mDelta, MatrixFloat &mNewDelta)
{
//    mNewDelta.resize(mDelta.rows(),mDelta.cols());
//    for(int i=0;i<mNewDelta.size();i++)
//        mNewDelta(i)=_pActivation->derivation(mInput(i))*mDelta(i);

//todo
}
///////////////////////////////////////////////////////////////////////////////
