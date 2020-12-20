/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#include "LayerActivation.h"

#include "Activations.h"

///////////////////////////////////////////////////////////////////////////////
LayerActivation::LayerActivation(const string& sActivation):
    Layer(sActivation)
{
    _pActivation=get_activation(sActivation);

	assert(_pActivation != nullptr);
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
void LayerActivation::predict(const MatrixFloat& mIn,MatrixFloat& mOut)
{
    mOut.resizeLike(mIn);

    for(Index i=0;i<mOut.size();i++)
        mOut(i)=_pActivation->apply(mIn(i));
}
///////////////////////////////////////////////////////////////////////////////
void LayerActivation::backpropagation(const MatrixFloat &mIn,const MatrixFloat &mGradientOut, MatrixFloat &mGradientIn)
{
	if (_bFirstLayer)
		return;

    mGradientIn.resizeLike(mIn);
    for(Index i=0;i<mGradientIn.size();i++)
        mGradientIn(i)=_pActivation->derivation(mIn(i));

    mGradientIn = mGradientIn.cwiseProduct(mGradientOut);
}
///////////////////////////////////////////////////////////////////////////////
