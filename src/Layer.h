/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#ifndef Layer_
#define Layer_

#include "Matrix.h"

#include <string>
using namespace std;

class Layer
{
public:
    Layer(const string& sType);
    virtual ~Layer();

    virtual Layer* clone() const =0;

	virtual void init();

    string type() const;

    int in_size() const;
    int out_size() const;
	void set_first_layer(bool bFirstLayer);

    virtual void forward(const MatrixFloat& mIn,MatrixFloat& mOut) =0;
	
    virtual void set_shape(int iInputRows,int iInputCols,int iInputPlanes,int& iOutputRows, int& iOutputCols,int & iOutputPlanes);
    virtual void backpropagation(const MatrixFloat &mIn,const MatrixFloat &mGradientOut, MatrixFloat &mGradientIn)=0;
	
	void set_train_mode(bool bTrainMode); //set to true to train, to false to test

    virtual bool has_weight() const;
    virtual MatrixFloat& weights();
    virtual MatrixFloat& gradient_weights();

protected:
    MatrixFloat _weight,_gradientWeight;
    int _iInputSize, _iOutputSize;
	int _iInputRows, _iInputCols, _iInputPlanes;
	int _iOutputRows, _iOutputCols, _iOutputPlanes;

	bool _bTrainMode;
	bool _bFirstLayer;

private:
    string _sType;
};

#endif
