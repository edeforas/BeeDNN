#ifndef Layer_
#define Layer_

#include "Matrix.h"

#include <string>
using namespace std;

class Layer
{
public:
    Layer(int iInSize, int iOutSize, string sType);
    virtual ~Layer();

    string type() const;
    int in_size() const;
    int out_size() const;

    virtual void forward(const MatrixFloat& mMatIn,MatrixFloat& mMatOut) const =0;
	
    virtual void init(); //init weight if any
    virtual void backpropagation(const MatrixFloat &mInput,const MatrixFloat &mDelta, float fLearningRate, MatrixFloat &mNewDelta)=0;
		
protected:
    int _iInSize, _iOutSize;

private:
    string _sType;
};

#endif
