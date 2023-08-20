/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#ifndef LayerChannelBias_
#define LayerChannelBias_

#include "Layer.h"
#include "Matrix.h"
namespace bee {
class LayerChannelBias : public Layer
{
public:
    explicit LayerChannelBias(Index iNbRows,Index iNbCols,Index iNbChannels, const std::string& sBiasInitializer = "Zeros");
    virtual ~LayerChannelBias();

    virtual Layer* clone() const override;
    virtual void init() override;

	void get_params(Index & iRows, Index & iCols, Index & iChannels) const;

    virtual void forward(const MatrixFloat& mIn, MatrixFloat &mOut) override;
    virtual void backpropagation(const MatrixFloat &mIn,const MatrixFloat &mGradientOut, MatrixFloat &mGradientIn) override;
private:
	Index _iNbRows,_iNbCols,_iNbChannels;
};
}
#endif
