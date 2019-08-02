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
    Layer(int iInSize, int iOutSize,const string& sType);
    virtual ~Layer();

    virtual Layer* clone() const =0;

    string type() const;

    void set_input_size(int iInputSize);
    int in_size() const;
    int out_size() const;

    virtual void forward(const MatrixFloat& mMatIn,MatrixFloat& mMatOut) const =0;
	
    virtual void init();
    virtual void backpropagation(const MatrixFloat &mInput,const MatrixFloat &mDelta, MatrixFloat &mNewDelta)=0;
		
	void set_train_mode(bool bTrainMode); //set to true to train, to false to test

    virtual bool has_weight();
    virtual MatrixFloat& weights();
    virtual MatrixFloat& gradient_weights();

protected:
    MatrixFloat _weight,_gradientWeight;
    int _iInSize, _iOutSize;
	bool _bTrainMode;

private:
    string _sType;
};

#endif
