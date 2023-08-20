/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#ifndef LayerMaxPool2D_
#define LayerMaxPool2D_

#include "Layer.h"
#include "Matrix.h"
namespace bee{
class LayerMaxPool2D : public Layer
{
public:
	explicit LayerMaxPool2D(Index iInRows, Index iInCols, Index iInChannels, Index iRowFactor = 2, Index iColFactor = 2);
    virtual ~LayerMaxPool2D() override;

	void get_params(Index& iInRows, Index& iInCols, Index& iInChannels, Index& iRowFactor, Index& iColFactor) const;

    virtual Layer* clone() const override;

    virtual void forward(const MatrixFloat& mIn, MatrixFloat &mOut) override;
    virtual void backpropagation(const MatrixFloat &mIn,const MatrixFloat &mGradientOut, MatrixFloat &mGradientIn) override;
	Index getOutputRows() const {
		return _iOutRows;
	}
	Index getOutputCols() const {
		return _iOutCols;
	}
private:
	Index _iInRows;
	Index _iInCols;
	Index _iInChannels;
	Index _iRowFactor;
	Index _iColFactor;
	Index _iOutRows;
	Index _iOutCols;

	Index _iInPlaneSize;
	Index _iOutPlaneSize;

	MatrixFloat _mMaxIndex;
};
}
#endif
