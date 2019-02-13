#ifndef LayerDenseNoBias_
#define LayerDenseNoBias_

#include "Layer.h"
#include "Matrix.h"

class LayerDenseNoBias : public Layer
{
public:
    LayerDenseNoBias(int iInSize,int iOutSize);
    virtual ~LayerDenseNoBias();
	
    virtual void forward(const MatrixFloat& mMatin, MatrixFloat &mMatOut) const;
	
    virtual void init();
    virtual void backpropagation(const MatrixFloat &mInput,const MatrixFloat &mDelta, float fLearningRate, MatrixFloat &mNewDelta);

    const MatrixFloat& weight() const;

private:
    MatrixFloat _weight;
};

#endif
