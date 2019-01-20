#ifndef Layer_
#define Layer_

#include "Matrix.h"

class Layer
{
public:
    Layer(int iInSize,int iOutSize);
    virtual ~Layer();

    virtual void forward(const MatrixFloat& mMatIn,MatrixFloat& mMatOut) const =0;

	//todo add void backpropagation();
	
    virtual void init_backpropagation() =0; //todo remove
    virtual void forward_save(const MatrixFloat& mMatIn,MatrixFloat& mMatOut) =0;     //todo remove
    virtual MatrixFloat get_weight_activation_derivation() const =0;     //todo  remove
    virtual MatrixFloat& get_weight() =0;     //todo  remove

    //learning variables, todo clean up / remove
    MatrixFloat in; //in todo remove
    MatrixFloat out; //fcn(in*weight)
    MatrixFloat outWeight; // in*weight todo remove
    MatrixFloat dE; //todo remove

protected:
    int _iInSize, _iOutSize;
};

#endif
