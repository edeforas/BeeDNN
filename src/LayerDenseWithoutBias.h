#ifndef LayerDenseWithoutBias_
#define LayerDenseWithoutBias_

#include "Layer.h"
#include "Matrix.h"

class LayerDenseWithoutBias : public Layer
{
public:
    LayerDenseWithoutBias(int iInSize,int iOutSize);
    virtual ~LayerDenseWithoutBias();
	
    virtual void forward(const MatrixFloat& mMatin, MatrixFloat &mMatOut) const;
	
    virtual void init();
    virtual void backpropagation(const MatrixFloat &mInput,const MatrixFloat &mDelta, float fLearningRate, MatrixFloat &mNewDelta);

private:
    MatrixFloat _weight;
};

#endif
