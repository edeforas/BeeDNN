/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#ifndef LayerRandomFlipLeftRight2D_
#define LayerRandomFlipLeftRight2D_

#include "Layer.h"
#include "Matrix.h"
#include <vector>

class LayerRandomFlipLeftRight2D : public Layer
{
public:
    explicit LayerRandomFlipLeftRight2D(Index iNbRows,Index iNbCols,Index iNbChannels);
    virtual ~LayerRandomFlipLeftRight2D();

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

#endif
