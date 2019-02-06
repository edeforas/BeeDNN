#ifndef Layer_
#define Layer_

#include "Matrix.h"

class Layer
{
public:
    Layer(int iInSize,int iOutSize);
    virtual ~Layer();

    virtual void forward(const MatrixFloat& mMatIn,MatrixFloat& mMatOut) const =0;
	
    virtual void init(); //init weight if any
    virtual void backpropagation(const MatrixFloat &mInput,const MatrixFloat &mDelta, float fLearningRate, MatrixFloat &mNewDelta)=0;

protected:
    int _iInSize, _iOutSize;
};

#endif
