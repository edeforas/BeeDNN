#ifndef _ConfusionMatrix_
#define _ConfusionMatrix_

#include "Matrix.h"

struct ClassificationResult
{
    double goodclassificationPercent;
    Matrix mConfMat;
};

class ConfusionMatrix
{
public:
    ClassificationResult compute(const Matrix& mRef, const Matrix& mTest, unsigned int iNbClass);
};

#endif
