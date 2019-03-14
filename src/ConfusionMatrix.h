#ifndef ConfusionMatrix_
#define ConfusionMatrix_

#include "Matrix.h"

struct ClassificationResult
{
    double accuracy;
    MatrixFloat mConfMat;
};

class ConfusionMatrix
{
public:
    ClassificationResult compute(const MatrixFloat& mRef, const MatrixFloat& mTest, int iNbClass=0);

    static void toPercent(const MatrixFloat& mConf, MatrixFloat& mConfPercent);
};

#endif
