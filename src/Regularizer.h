/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#ifndef Regularizer_
#define Regularizer_

#include "Matrix.h"

#include <vector>
#include <string>
namespace bee {
class Layer;
class Regularizer
{
public:
	Regularizer();
	virtual ~Regularizer();
	
	virtual void set_parameter(float fParameter);
	float get_parameter() const;

	virtual std::string name() const = 0;

    virtual void apply(MatrixFloat& w,MatrixFloat& dw) = 0;

protected:
	float _fParameter;
};

Regularizer* create_regularizer(const std::string & sRegularizer);
void list_regularizer_available(std::vector<std::string>& vsRegularizers);
}
#endif
