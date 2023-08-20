/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#ifndef LayerConvolution2D_
#define LayerConvolution2D_

#include <vector>
using namespace std;

#include "Layer.h"
#include "Matrix.h"

namespace bee {
class LayerConvolution2D : public Layer
{
public:
	LayerConvolution2D(Index iInRows, Index iInCols,Index iInChannels, Index iKernelRows, Index iKernelCols,Index iOutChannels,Index iRowStride=1, Index iColStride=1);
    virtual ~LayerConvolution2D() override;

	void get_params(Index& iInRows, Index& iInCols,Index& iInChannels, Index& iKernelRows, Index& iKernelCols,Index& iOutChannels) const;

    virtual Layer* clone() const override;

	virtual void init() override;

    void get_params(Index & iInRows, Index & iInCols, Index & iInChannels, Index & iKernelRows, Index & iKernelCols, Index & iOutChannels, Index & iRowStride, Index & iColStride) const;

    virtual void forward(const MatrixFloat& mIn, MatrixFloat &mOut) override;
    virtual void backpropagation(const MatrixFloat &mIn,const MatrixFloat &mGradientOut, MatrixFloat &mGradientIn) override;

	//public for tests
	void im2col(const MatrixFloat & mIn, MatrixFloat & mCol);
	void im2col_LUT(const MatrixFloat & mIn, MatrixFloat & mCol);
	void col2im(const MatrixFloat & mCol, MatrixFloat & mIm);
	void col2im_LUT(const MatrixFloat & mCol, MatrixFloat & mIm);

	bool fastLUT; //temporary
	Index getOutputRows() const {
		return _iOutRows;
	}
		Index getOutputCols() const {
		return _iOutCols;
	}
private:
	void reshape_to_out(MatrixFloat & mOut);
	void reshape_from_out(MatrixFloat & mOut);
	
	// LUT algo
	void create_im2col_LUT();
	vector<Index> _im2ColLUT;

	Index _iInRows;
	Index _iInCols;
	Index _iSamples;
	Index _iInChannels;
	Index _iKernelRows;
	Index _iKernelCols;
	Index _iRowStride;
	Index _iColStride;
	Index _iOutChannels;
	Index _iBorderRows;
	Index _iBorderCols;
	Index _iOutRows;
	Index _iOutCols;

public:
	MatrixFloat _im2colT; // input image, im2col format
	MatrixFloat _tempImg; // temporary image, to avoid malloc
};
}
#endif
