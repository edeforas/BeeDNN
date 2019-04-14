/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#ifndef Loss_
#define Loss_

#include "Matrix.h"

#include <string>
#include <vector>
using namespace std;

class Loss
{
public:
    Loss();
    virtual ~Loss();

    virtual string name() const =0 ;

	virtual float compute(const MatrixFloat& mPredicted,const MatrixFloat& mTarget) const =0 ;
	virtual float compute_derivate(const MatrixFloat& mPredicted,const MatrixFloat& mTarget) const =0 ;
};

Loss* get_loss(const string & sLoss);
void list_loss_available(vector<string>& vsLoss);

#endif
