/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#ifndef Metrics_
#define Metrics_

#include "Matrix.h"
namespace bee{
class Metrics
{
public:
    Metrics();

    void compute(const MatrixFloat& mRef, const MatrixFloat& mTest, Index iNbClass=0); //if iNbClass==0 guess using max index

    float accuracy() const;
    float balanced_accuracy() const;

    MatrixFloat confusion_matrix() const; // raw count
    MatrixFloat confusion_matrix_normalized() const; // in %

private:
    float _accuracy;
    float _balancedAccuracy;
    MatrixFloat _mConfusionMatrix; //raw count
    MatrixFloat _mConfusionMatrixNormalized; //in %
};
}
#endif
