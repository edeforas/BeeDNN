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
	
    virtual void forward(const MatrixFloat& mIn, MatrixFloat &mOut) const override;
	
    virtual void backpropagation(const MatrixFloat &mInput,const MatrixFloat &mDelta, float fLearningRate, MatrixFloat &mNewDelta) override;

	virtual void to_string(string& sBuffer);
private:
    Activation * _pActivation;
};

#endif
