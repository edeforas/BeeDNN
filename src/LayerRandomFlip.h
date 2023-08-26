/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#pragma once

#include "Layer.h"
#include "Matrix.h"
#include <vector>
namespace beednn {
// only left-right (horizontal) mode for now
class LayerRandomFlip : public Layer
{
public:
    explicit LayerRandomFlip(Index iNbRows,Index iNbCols,Index iNbChannels);
    virtual ~LayerRandomFlip();

    virtual Layer* clone() const override;
    virtual void init() override;

	void get_params(Index & iRows, Index & iCols, Index & iChannels) const;

    virtual void forward(const MatrixFloat& mIn, MatrixFloat &mOut) override;
    virtual void backpropagation(const MatrixFloat &mIn,const MatrixFloat &mGradientOut, MatrixFloat &mGradientIn) override;
private:
	Index _iNbRows,_iNbCols,_iNbChannels;
    Index _iPlaneSize;
    MatrixFloat _flipped;
};
}
