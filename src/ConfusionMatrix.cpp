#include "ConfusionMatrix.h"

///////////////////////////////////////////////////////////////////////////////
ClassificationResult ConfusionMatrix::compute(const MatrixFloat& mRef,const MatrixFloat& mTest,int iNbClass)
{
    if(iNbClass==0)
        iNbClass=(int)mRef.maxCoeff()+1; //guess the nb of class

    ClassificationResult cr;
    cr.mConfMat.resize(iNbClass,iNbClass);
    cr.mConfMat.setZero();

    for(unsigned int i=0;i<(unsigned int)mRef.rows();i++)
    {
        //threshold label
        int iLabel=(unsigned int)(mTest(i)+0.5f);
        iLabel=std::min(iLabel,iNbClass-1);
        iLabel=std::max(iLabel,0);
        cr.mConfMat((unsigned int)mRef(i),iLabel)++;
    }

    MatrixFloat mDiag=cr.mConfMat.diagonal();
    MatrixFloat mSum=rowWiseSum(cr.mConfMat);

    MatrixFloat mGoodClassification=mDiag.cwiseQuotient(mSum)*100.f;

    cr.accuracy=mGoodClassification.sum()/iNbClass;

    return cr;
}
///////////////////////////////////////////////////////////////////////////////
void ConfusionMatrix::toPercent(const MatrixFloat& mConf, MatrixFloat& mConfPercent)
{
    MatrixFloat mSumRow=rowWiseSum(mConf);
    mConfPercent=rowWiseDivide(mConf,mSumRow)*100.f;
}
///////////////////////////////////////////////////////////////////////////////

