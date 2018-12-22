#ifndef ActivationLayer_
#define ActivationLayer_

#include "Layer.h"
#include "Matrix.h"

#include <string>
using namespace std;

class Activation;

class ActivationLayer : public Layer
{
public:
    ActivationLayer(int iInSize,int iOutSize, string sActivation);
    ~ActivationLayer();
    virtual void init();

    virtual void forward(const MatrixFloat& mMatin, MatrixFloat &mMatOut) const;
    virtual void forward_save(const MatrixFloat& mMatin, MatrixFloat &mMatOut); //todo remove

    virtual MatrixFloat get_weight_activation_derivation() const;
    virtual MatrixFloat& get_weight();

private:
    MatrixFloat _weight;
    const Activation* _pActiv;
};

#endif
