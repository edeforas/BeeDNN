/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#ifndef LayerConvolution2D_
#define LayerConvolution2D_

#include "Layer.h"
#include "Matrix.h"

class LayerConvolution2D : public Layer
{
public:
	LayerConvolution2D(int iInRows, int iInCols,int iInPlanes, int iKernelRows, int iKernelCols,int  iOutPlanes);
    virtual ~LayerConvolution2D() override;

	void get_params(int& iInRows, int& iInCols,int& iInPlanes, int& iKernelRows, int& iKernelCols,int& iOutPlanes);

    virtual Layer* clone() const override;

    virtual void forward(const MatrixFloat& mIn, MatrixFloat &mOut) override;
    virtual void backpropagation(const MatrixFloat &mIn,const MatrixFloat &mGradientOut, MatrixFloat &mGradientIn) override;

private:
	void conv_and_add(const MatrixFloat& mImage,const MatrixFloat& mKernel, MatrixFloat&mOut);

	int _iInRows;
	int _iInCols;
	int _iInPlanes;
	int _iKernelRows;
	int _iKernelCols;
	int _iOutPlanes;
	int _iBorderRows;
	int _iBorderCols;
	int _iOutRows;
	int _iOutCols;
};

#endif
