#ifndef Layer_
#define Layer_

#include "Matrix.h"

class Layer
{
public:
    Layer();
    virtual ~Layer();
    virtual void init() =0;

    virtual void forward(const Matrix& mMatIn,Matrix& mMatOut) const =0;
    virtual void forward_save(const Matrix& mMatIn,Matrix& mMatOut) =0;

    virtual Matrix get_weight_activation_derivation() =0;
    virtual Matrix& get_weight() =0;

    //learning variables, todo clean up / remove
    Matrix in; //in
    Matrix out; //fcn(in*weight)
    Matrix outWeight; // in*weight
    Matrix dE;
};

#endif
