#include "ConfusionMatrix.h"
///////////////////////////////////////////////////////////////////////////////
ClassificationResult ConfusionMatrix::compute(const MatrixFloat& mRef,const MatrixFloat& mTest,unsigned int iNbClass)
{
    ClassificationResult cr;
    cr.mConfMat.resize(iNbClass,iNbClass);
    cr.mConfMat.setZero();

    for(unsigned int i=0;i<mRef.rows();i++)
    {
        cr.mConfMat((unsigned int)mRef(i),(unsigned int)mTest(i))++;
    }

    MatrixFloat mDiag=cr.mConfMat.diag();
    MatrixFloat mSum=cr.mConfMat.row_sum();

    MatrixFloat mGoodClassification=mDiag.cwiseDivide(mSum)*100.f;

    cr.goodclassificationPercent=mGoodClassification.sum()/iNbClass;

    return cr;
}
///////////////////////////////////////////////////////////////////////////////
