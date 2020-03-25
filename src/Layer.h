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

    string type() const;

	void set_first_layer(bool bFirstLayer);

    virtual void forward(const MatrixFloat& mIn,MatrixFloat& mOut) =0;
	
    virtual void init();
    virtual void backpropagation(const MatrixFloat &mIn,const MatrixFloat &mGradientOut, MatrixFloat &mGradientIn)=0;
	
	void set_train_mode(bool bTrainMode); //set to true to train, to false to test

    bool has_weight() const;
    MatrixFloat& weights();
    MatrixFloat& gradient_weights();

	bool has_bias() const;
	MatrixFloat& bias();
	MatrixFloat& gradient_bias();

protected:
    MatrixFloat _weight,_gradientWeight;
	MatrixFloat _bias, _gradientBias;
	bool _bTrainMode;
	bool _bFirstLayer;

private:
    string _sType;
};

#endif
