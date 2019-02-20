#ifndef LayerDenseAndBias_
#define LayerDenseAndBias_

#include "Layer.h"
#include "Matrix.h"

class LayerDenseAndBias : public Layer
{
public:
    LayerDenseAndBias(int iInSize,int iOutSize);
    virtual ~LayerDenseAndBias();
	
    virtual void forward(const MatrixFloat& mMatin, MatrixFloat &mMatOut) const override;
	
    virtual void init() override;
    virtual void backpropagation(const MatrixFloat &mInput,const MatrixFloat &mDelta, float fLearningRate, MatrixFloat &mNewDelta) override;

    const MatrixFloat& weight() const;
    const MatrixFloat& bias() const;

private:
    MatrixFloat _weight;
    MatrixFloat _bias;
};

#endif
