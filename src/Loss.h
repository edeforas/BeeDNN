/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/
#pragma once

#include "Matrix.h"

#include <string>
#include <vector>
namespace beednn {
class Loss
{
public:
    Loss();
    virtual ~Loss();

    virtual std::string name() const =0 ;

	virtual void compute(const MatrixFloat& mPredicted, const MatrixFloat& mTarget, MatrixFloat& mLoss) const = 0;
	virtual void compute_gradient(const MatrixFloat& mPredicted,const MatrixFloat& mTarget, MatrixFloat& mGradientLoss) const =0 ;

    // loss weight class balancing, one row by class, ideal value = 1.f
	// as in: https://towardsdatascience.com/handling-imbalanced-datasets-in-deep-learning-f48407a0e758 (chapter (1) )
	// for no balancing, set to empty matrix (default)
    void set_class_balancing(const MatrixFloat& mWeight);

protected:
    void balance_with_weight(const MatrixFloat& mTruth, MatrixFloat& mGradient) const;

    MatrixFloat _mWeightBalancing;
	bool _bClassBalancing;
};

Loss* create_loss(const std::string & sLoss);
void list_loss_available(std::vector<std::string>& vsLoss);
}

