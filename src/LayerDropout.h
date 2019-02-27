#ifndef LayerDropout_
#define LayerDropout_

#include "Layer.h"
#include "Matrix.h"

#include <string>
using namespace std;

class Activation;

class LayerDropout : public Layer
{
public:
    LayerDropout(int iSize,float fRate);
    virtual ~LayerDropout() override;
	
    virtual void forward(const MatrixFloat& mIn, MatrixFloat &mOut) const override;
	
    virtual void backpropagation(const MatrixFloat &mInput,const MatrixFloat &mDelta, float fLearningRate, MatrixFloat &mNewDelta) override;

	virtual void to_string(string& sBuffer);

private:
	void create_mask(int iSize);

	float _fRate;
	MatrixFloat _mask;
};

#endif
