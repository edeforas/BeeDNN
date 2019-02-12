#ifndef LayerDenseAndBias_
#define LayerDenseAndBias_

#include "Layer.h"
#include "Matrix.h"

class LayerDenseAndBias : public Layer
{
public:
    LayerDenseAndBias(int iInSize,int iOutSize);
    virtual ~LayerDenseAndBias();
	
    virtual void forward(const MatrixFloat& mMatin, MatrixFloat &mMatOut) const;
	
    virtual void init();
    virtual void backpropagation(const MatrixFloat &mInput,const MatrixFloat &mDelta, float fLearningRate, MatrixFloat &mNewDelta);

    virtual void to_string(string& sBuffer);

private:
    MatrixFloat _weight;
    MatrixFloat _bias;
};

#endif
