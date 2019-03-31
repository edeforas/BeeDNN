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

class Layer;

class Optimizer
{
public:
	Optimizer();
	virtual ~Optimizer();
	
	virtual void init(const Layer& l)=0;

	virtual void optimize(MatrixFloat& weight, const MatrixFloat& mDx) = 0;

	float fLearningRate; //temp
	float fMomentum; //temp
    float fDecay; //temp
};

Optimizer* get_optimizer(const string & sOptimizer);
void list_optimizers_available(vector<string>& vsOptimizers);

#endif
