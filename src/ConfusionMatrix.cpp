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

    MatrixFloat mDiag=cr.mConfMat.diagonal();
    MatrixFloat mSum=rowWiseSum(cr.mConfMat);

    MatrixFloat mGoodClassification=mDiag.cwiseQuotient(mSum)*100.f;

    cr.goodclassificationPercent=mGoodClassification.sum()/iNbClass;

    return cr;
}
///////////////////////////////////////////////////////////////////////////////
