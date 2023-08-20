/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#include "LayerActivation.h"

#include "Activations.h"
namespace bee{

using namespace std;
///////////////////////////////////////////////////////////////////////////////
LayerActivation::LayerActivation(const string& sActivation):
    Layer(sActivation)
{
    _pActivation=get_activation(sActivation);

	assert(_pActivation);
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
void LayerActivation::forward(const MatrixFloat& mIn,MatrixFloat& mOut)
{
    assert(_pActivation);
    mOut.resizeLike(mIn);

    for(Index i=0;i<mOut.size();i++)
        mOut(i)=_pActivation->apply(mIn(i));
}
///////////////////////////////////////////////////////////////////////////////
void LayerActivation::backpropagation(const MatrixFloat &mIn,const MatrixFloat &mGradientOut, MatrixFloat &mGradientIn)
{
    assert(mIn.rows() == mGradientOut.rows());
    assert(mIn.cols() == mGradientOut.cols());
    assert(_pActivation);

	if (_bFirstLayer)
		return;

    mGradientIn.resizeLike(mGradientOut);
    for(Index i=0;i<mGradientIn.size();i++)
        mGradientIn(i)=_pActivation->derivation(mIn(i));

    mGradientIn = mGradientIn.cwiseProduct(mGradientOut);
}
///////////////////////////////////////////////////////////////////////////////
}