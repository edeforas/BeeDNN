/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#include "ConfusionMatrix.h"

#include <cmath>
#include <cassert>

///////////////////////////////////////////////////////////////////////////////
ClassificationResult ConfusionMatrix::compute(const MatrixFloat& mRef,const MatrixFloat& mTest,int iNbClass)
{
	assert(mRef.rows() == mTest.rows());

    if(iNbClass==0)
        iNbClass=(int)mRef.maxCoeff()+1; //guess the number of class

    ClassificationResult cr;
    cr.mConfMat.setZero(iNbClass, iNbClass);

    for(unsigned int i=0;i<(unsigned int)mRef.rows();i++)
    {
        //threshold label
        int iLabelPredicted=(int)(std::roundf(mTest(i)));
		iLabelPredicted =std::min(iLabelPredicted,iNbClass-1);
		iLabelPredicted =std::max(iLabelPredicted,0);
        cr.mConfMat((unsigned int)mRef(i), iLabelPredicted)++;
    }

    //compute accuracy in percent
    float fTrace=cr.mConfMat.trace();
    float fSum=cr.mConfMat.sum();
    cr.accuracy=fTrace/fSum*100.f;

    return cr;
}
///////////////////////////////////////////////////////////////////////////////
void ConfusionMatrix::toPercent(const MatrixFloat& mConf, MatrixFloat& mConfPercent)
{
    MatrixFloat mSumRow=rowWiseSum(mConf);
    mConfPercent=rowWiseDivide(mConf,mSumRow)*100.f;
}
///////////////////////////////////////////////////////////////////////////////
