/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#ifndef Optimizer_
#define Optimizer_

#include "Matrix.h"

#include <vector>
#include <string>
using namespace std;

namespace bee{
class Layer;

class Optimizer
{
public:
	Optimizer();
	virtual ~Optimizer();
	
	virtual Optimizer* clone() = 0;

	void set_learningrate(float fLearningRate);  //-1.f is for default
	float get_learningrate();  //-1.f is for default

	void set_params(float fLearningRate = -1.f, float fDecay = -1.f, float fMomentum = -1.f);  //-1.f is for default params

	virtual string name() const = 0;
	virtual void init()=0;

    virtual void optimize(MatrixFloat& w, const MatrixFloat& dw) = 0;

protected:
	float _fLearningRate;
	float _fMomentum;
    float _fDecay;
};

Optimizer* create_optimizer(const string & sOptimizer);
void list_optimizers_available(vector<string>& vsOptimizers);
}
#endif
