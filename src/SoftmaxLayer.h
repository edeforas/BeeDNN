#ifndef SoftmaxLayer_
#define SoftmaxLayer_

#include "Layer.h"
#include "Matrix.h"

class Activation;

class SoftmaxLayer : public Layer
{
public:
    SoftmaxLayer();
    ~SoftmaxLayer();
    virtual void init();

    virtual void forward(const Matrix& mMatin, Matrix &mMatOut) const;
    void backward(const Matrix& mErrorIn,Matrix& mErrorOut);

    virtual void forward_save(const Matrix& mMatin, Matrix &mMatOut);

    virtual Matrix get_weight_activation_derivation();
    virtual Matrix& get_weight();

};

#endif
