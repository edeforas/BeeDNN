#include "ConfusionMatrix.h"
///////////////////////////////////////////////////////////////////////////////
ClassificationResult ConfusionMatrix::compute(const MatrixFloat& mRef,const MatrixFloat& mTest,unsigned int iNbClass)
{
    ClassificationResult cr;
    cr.mConfMat.resize(iNbClass,iNbClass);
    cr.mConfMat.set_zero();

    for(unsigned int i=0;i<mRef.rows();i++)
    {
        cr.mConfMat((unsigned int)mRef(i),(unsigned int)mTest(i))++;
    }

    MatrixFloat mDiag=cr.mConfMat.diag();
    MatrixFloat mSum=cr.mConfMat.row_sum();

    MatrixFloat mGoodClassification=mDiag.element_divide(mSum)*100.;

    cr.goodclassificationPercent=mGoodClassification.sum()/iNbClass;

    return cr;
}
///////////////////////////////////////////////////////////////////////////////
