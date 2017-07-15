#ifndef DenseLayer_
#define DenseLayer_

#include "Layer.h"
#include "Matrix.h"

class Activation;

class DenseLayer : public Layer
{
public:
    DenseLayer(int iInSize,int iOutSize,const Activation* activ);
    ~DenseLayer();
    virtual void init();

    virtual void forward(const Matrix& mMatin, Matrix &mMatOut) const;
    virtual void forward_save(const Matrix& mMatin, Matrix &mMatOut);

    virtual Matrix get_weight_activation_derivation();
    virtual Matrix& get_weight();

private:
    Matrix _weight;
    const Activation* _activ;
    int _iInSize, _iOutSize;
};

#endif
