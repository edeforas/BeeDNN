/*
    Copyright (c) 2021, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#include "LayerTimeDistributedBias.h"
#include "Initializers.h"
using namespace std;
namespace beednn {

///////////////////////////////////////////////////////////////////////////////
LayerTimeDistributedBias::LayerTimeDistributedBias(int iFrameSize,const string& sBiasInitializer ) :
    Layer("TimeDistributedBias")
{
	_iFrameSize=iFrameSize;
    set_bias_initializer(sBiasInitializer);
    LayerTimeDistributedBias::init();
}
///////////////////////////////////////////////////////////////////////////////
LayerTimeDistributedBias::~LayerTimeDistributedBias()
{ }
///////////////////////////////////////////////////////////////////////////////
Layer* LayerTimeDistributedBias::clone() const
{
    LayerTimeDistributedBias* pLayer=new LayerTimeDistributedBias(_iFrameSize, bias_initializer());
	pLayer->_bias = _bias;

    return pLayer;
}
///////////////////////////////////////////////////////////////////////////////
int LayerTimeDistributedBias::frame_size() const
{
    return _iFrameSize;
}
///////////////////////////////////////////////////////////////////////////////
void LayerTimeDistributedBias::init()
{
    Initializers::compute(bias_initializer(), _bias, 1, _iFrameSize);
    Layer::init();
}
///////////////////////////////////////////////////////////////////////////////
void LayerTimeDistributedBias::forward(const MatrixFloat& mIn,MatrixFloat& mOut)
{
    // reshape the input to (x, _iFrameSize), compute, reshape back
    MatrixFloat mInR = viewResize(mIn,mIn.size()/ _iFrameSize,_iFrameSize);
    mOut=rowWiseAdd(mInR, _bias);
    mOut.resize(mIn.rows(), mIn.cols());
}
///////////////////////////////////////////////////////////////////////////////
void LayerTimeDistributedBias::backpropagation(const MatrixFloat &mIn,const MatrixFloat &mGradientOut, MatrixFloat &mGradientIn)
{
    (void)mIn;

    // reshape the gradient to (x, _iFrameSize), compute
    MatrixFloat mGradientOutR = viewResize(mGradientOut, mGradientOut.size() / _iFrameSize, _iFrameSize);
    _gradientBias = colWiseMean(mGradientOutR);

	if (_bFirstLayer)
		return;

    mGradientIn = mGradientOut;
}
///////////////////////////////////////////////////////////////////////////////
}