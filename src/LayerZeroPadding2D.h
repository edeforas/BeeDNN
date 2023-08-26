/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#ifndef LayerZeroPadding2D_
#define LayerZeroPadding2D_

#include "Layer.h"
#include "Matrix.h"

// ZeroPadding2D as in : https://www.tensorflow.org/api_docs/python/tf/keras/layers/ZeroPadding2D 
// and
// https://deeplizard.com/learn/video/qSTv_m-KFk0
namespace beednn {
class LayerZeroPadding2D : public Layer
{
public:
	explicit LayerZeroPadding2D(Index iInRows, Index iInCols, Index iInChannels, Index iBorder=1);
    virtual ~LayerZeroPadding2D() override;

	void get_params(Index& iInRows, Index& iInCols, Index& iInChannels, Index& iBorder) const;

    virtual Layer* clone() const override;

    virtual void forward(const MatrixFloat& mIn, MatrixFloat &mOut) override;
    virtual void backpropagation(const MatrixFloat &mIn,const MatrixFloat &mGradientOut, MatrixFloat &mGradientIn) override;

private:
	Index _iInRows;
	Index _iInCols;
	Index _iInChannels;
	Index _iBorder;
};
}
#endif
