/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#include "LayerActivation.h"

#include "Activation.h"

///////////////////////////////////////////////////////////////////////////////
LayerActivation::LayerActivation(const string& sActivation):
    Layer(0,0,sActivation)
{
    _pActivation=get_activation(sActivation);
}
///////////////////////////////////////////////////////////////////////////////
LayerActivation::~LayerActivation()
{
	delete _pActivation;
}
///////////////////////////////////////////////////////////////////////////////
Layer* LayerActivation::clone() const
{
    return new LayerActivation(_pActivation->name());
}
///////////////////////////////////////////////////////////////////////////////
void LayerActivation::forward(const MatrixFloat& mIn,MatrixFloat& mOut) const
{
    mOut.resize(mIn.rows(),mIn.cols());

    for(int i=0;i<mOut.size();i++)
        mOut(i)=_pActivation->apply(mIn(i));
}
///////////////////////////////////////////////////////////////////////////////
void LayerActivation::backpropagation(const MatrixFloat &mInput,const MatrixFloat &mDelta, Optimizer* pOptim, MatrixFloat &mNewDelta)
{
    (void)pOptim;

    mNewDelta.resize(mDelta.rows(),mDelta.cols());
    for(int i=0;i<mNewDelta.size();i++)
        mNewDelta(i)=_pActivation->derivation(mInput(i))*mDelta(i);
}
///////////////////////////////////////////////////////////////////////////////
