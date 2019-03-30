#ifndef LayerGlobalGain_
#define LayerGlobalGain_

#include "Layer.h"
#include "Matrix.h"

class LayerGlobalGain : public Layer
{
public:
    LayerGlobalGain(int iInSize,float fGlobalGain);
    virtual ~LayerGlobalGain();

    virtual Layer* clone() const override;

    virtual void init() override;
    virtual void forward(const MatrixFloat& mMatin, MatrixFloat &mMatOut) const override;

    virtual void backpropagation(const MatrixFloat &mInput,const MatrixFloat &mDelta, Optimizer* pOptim, MatrixFloat &mNewDelta) override;

    float gain() const;
    bool is_learned() const;

private:
    MatrixFloat _globalGain,_mDx;
//    float _fGlobalGain;
    bool _bLearnable;
};

#endif
