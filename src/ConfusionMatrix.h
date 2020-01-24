/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#ifndef ConfusionMatrix_
#define ConfusionMatrix_

#include "Matrix.h"

struct ClassificationResult
{
    float accuracy;
    MatrixFloat mConfMat;
};

class ConfusionMatrix
{
public:
    ClassificationResult compute(const MatrixFloat& mRef, const MatrixFloat& mTest, Index iNbClass=0);

    static void toPercent(const MatrixFloat& mConf, MatrixFloat& mConfPercent);
};

#endif
