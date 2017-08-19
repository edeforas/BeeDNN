#ifndef _ConfusionMatrix_
#define _ConfusionMatrix_

#include "Matrix.h"

struct ClassificationResult
{
    double goodclassificationPercent;
    MatrixFloat mConfMat;
};

class ConfusionMatrix
{
public:
    ClassificationResult compute(const MatrixFloat& mRef, const MatrixFloat& mTest, unsigned int iNbClass);
};

#endif
