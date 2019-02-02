#ifndef LayerDenseWithBias_
#define LayerDenseWithBias_

#include "Layer.h"
#include "Matrix.h"

class LayerDenseWithBias : public Layer
{
public:
    LayerDenseWithBias(int iInSize,int iOutSize);
    virtual ~LayerDenseWithBias();
	
    virtual void forward(const MatrixFloat& mMatin, MatrixFloat &mMatOut) const;
	
    virtual void init();
    virtual void backpropagation(const MatrixFloat &mInput,const MatrixFloat &mDelta, float fLearningRate, MatrixFloat &mNewDelta);

private:
    MatrixFloat _weight;
    MatrixFloat _bias;
};

#endif
