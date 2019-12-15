/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#ifndef LayerPoolMax2D_
#define LayerPoolMax2D_

#include "Layer.h"
#include "Matrix.h"

class LayerPoolMax2D : public Layer
{
public:
	LayerPoolMax2D(int iRowFactor = 2, int iColFactor = 2);
    virtual ~LayerPoolMax2D() override;

	virtual void init() override;
	
	void get_params(int& iRowFactor, int& iColFactor);

    virtual Layer* clone() const override;

    virtual void forward(const MatrixFloat& mIn, MatrixFloat &mOut) override;
    virtual void backpropagation(const MatrixFloat &mIn,const MatrixFloat &mGradientOut, MatrixFloat &mGradientIn) override;

private:
	int _iRowFactor;
	int _iColFactor;
	int _iOutRows;
	int _iOutCols;

	int _iInPlaneSize;
	int _iOutPlaneSize;

	MatrixFloat _mMaxIndex;
};

#endif
