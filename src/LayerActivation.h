#ifndef LayerActivation_
#define LayerActivation_

#include "Layer.h"
#include "Matrix.h"

#include <string>
using namespace std;

class Activation;

class LayerActivation : public Layer
{
public:
    LayerActivation(const string& sActivation);
    virtual ~LayerActivation() override;

    virtual Layer* clone() const override;

    virtual void forward(const MatrixFloat& mIn, MatrixFloat &mOut) const override;
	
    virtual void backpropagation(const MatrixFloat &mInput,const MatrixFloat &mDelta, Optimizer* pOptim, MatrixFloat &mNewDelta) override;

private:
    Activation * _pActivation;
};

#endif
