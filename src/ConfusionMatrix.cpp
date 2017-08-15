#include "ConfusionMatrix.h"


ClassificationResult ConfusionMatrix::compute(Matrix mRef,Matrix mTest,unsigned int iNbClass)
{
    ClassificationResult cr;
    cr.mConfMat.resize(iNbClass,iNbClass);
    cr.mConfMat.set_zero();

    for(unsigned int i=0;i<mRef.rows();i++)
    {
        cr.mConfMat((unsigned int)mRef(i),(unsigned int)mTest(i))++;
    }

    Matrix mDiag=cr.mConfMat.diag();
    Matrix mSum=cr.mConfMat.row_sum();

    Matrix mGoodClassification=mDiag.element_divide(mSum)*100.;

    cr.goodclassificationPercent=mGoodClassification.sum()/iNbClass;

    return cr;
}



