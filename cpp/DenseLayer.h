#ifndef DenseLayer_
#define DenseLayer_

#include "Layer.h"
#include "Activation.h"
#include "Matrix.h"

class DenseLayer : public Layer
{
public:
    DenseLayer(int iInSize,int iOutSize,const Activation& activ);

    virtual void forward(const Matrix& mMatin, Matrix &mMatOut) const;
    virtual void forward_feed(const Matrix& mMatin, Matrix &mMatOut);

    virtual void init_weight();
    virtual void init_DE();
    virtual Matrix get_weight_activation_derivation();
    virtual Matrix& get_weight();

private:
    Matrix _weight;
    const Activation& _activ;
    int _iInSize, _iOutSize;
};

#endif
