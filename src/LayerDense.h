#ifndef LayerDense_
#define LayerDense_

#include "Layer.h"
#include "Matrix.h"

class LayerDense : public Layer
{
public:
    LayerDense(int iInSize,int iOutSize,bool bHasBias);
    virtual ~LayerDense();
	
    virtual void forward(const MatrixFloat& mMatin, MatrixFloat &mMatOut) const override;
	
    virtual void init() override;
    virtual void backpropagation(const MatrixFloat &mInput,const MatrixFloat &mDelta, Optimizer* pOptim, MatrixFloat &mNewDelta) override;

	bool has_bias() const;
    const MatrixFloat& weight() const;
    const MatrixFloat& bias() const;

private:
    MatrixFloat _weight;
    MatrixFloat _bias;
	bool _bHasBias;
};

#endif
