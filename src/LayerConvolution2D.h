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
	LayerConvolution2D(int iInRows, int iInCols,int iInChannels, int iKernelRows, int iKernelCols,int iOutChannels);
    virtual ~LayerConvolution2D() override;

	void get_params(int& iInRows, int& iInCols,int& iInChannels, int& iKernelRows, int& iKernelCols,int& iOutChannels);

    virtual Layer* clone() const override;

	virtual void init() override;

    virtual void forward(const MatrixFloat& mIn, MatrixFloat &mOut) override;
    virtual void backpropagation(const MatrixFloat &mIn,const MatrixFloat &mGradientOut, MatrixFloat &mGradientIn) override;

	void im2col(const MatrixFloat & mIn, MatrixFloat & mCol);
	void col2im(const MatrixFloat & mCol, MatrixFloat & mIm);

private:
	void reshape_to_out(MatrixFloat & mOut);
	void reshape_from_out(MatrixFloat & mOut);

	int _iInRows;
	int _iInCols;
	int _iSamples;
	int _iInChannels;
	int _iKernelRows;
	int _iKernelCols;
	int _iOutChannels;
	int _iBorderRows;
	int _iBorderCols;
	int _iOutRows;
	int _iOutCols;

	MatrixFloat _im2col; // input image, im2col format
	MatrixFloat _tempImg; // temporary image, to avoid malloc
};

#endif
