#include "ConfusionMatrix.h"

///////////////////////////////////////////////////////////////////////////////
ClassificationResult ConfusionMatrix::compute(const MatrixFloat& mRef,const MatrixFloat& mTest,unsigned int iNbClass)
{
    if(iNbClass==0)
        iNbClass=(int)mRef.maxCoeff()+1; //guess the nb of class

    ClassificationResult cr;
    cr.mConfMat.resize(iNbClass,iNbClass);
    cr.mConfMat.setZero();

    for(unsigned int i=0;i<(unsigned int)mRef.rows();i++)
    {
        cr.mConfMat((unsigned int)mRef(i),(unsigned int)(mTest(i)+0.5f))++;
    }

    MatrixFloat mDiag=cr.mConfMat.diagonal();
    MatrixFloat mSum=rowWiseSum(cr.mConfMat);

    MatrixFloat mGoodClassification=mDiag.cwiseQuotient(mSum)*100.f;

    cr.accuracy=mGoodClassification.sum()/iNbClass;

    return cr;
}
///////////////////////////////////////////////////////////////////////////////
