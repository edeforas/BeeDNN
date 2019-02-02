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
    LayerActivation(string sActivation);
    virtual ~LayerActivation();
	
    virtual void forward(const MatrixFloat& mIn, MatrixFloat &mOut) const;
	
    virtual void backpropagation(const MatrixFloat &mInput,const MatrixFloat &mDelta, float fLearningRate, MatrixFloat &mNewDelta);

private:
    Activation * _pActivation;
};

#endif
