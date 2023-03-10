/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#include "Metrics.h"

#include <cmath>
#include <cassert>

///////////////////////////////////////////////////////////////////////////////
Metrics::Metrics()
{
    _accuracy = 0.f;
    _balancedAccuracy = 0.f;
}
///////////////////////////////////////////////////////////////////////////////
float Metrics::accuracy() const
{
    return _accuracy;
}

float Metrics::balanced_accuracy() const
{
    return _balancedAccuracy;
}

MatrixFloat Metrics::confusion_matrix() const
{
    return _mConfusionMatrix;
}

MatrixFloat Metrics::confusion_matrix_normalized() const
{
    return _mConfusionMatrixNormalized;
}


void Metrics::compute(const MatrixFloat& mRef,const MatrixFloat& mTest, Index iNbClass)
{
	// some formula from https://www.arxiv-vanity.com/papers/2008.05756/
	
	assert(mRef.rows() == mTest.rows());

    if(iNbClass==0)
        iNbClass=(Index)mRef.maxCoeff()+1; //guess the number of class

    _mConfusionMatrix.setZero(iNbClass, iNbClass);

    for(Index i=0;i<mRef.rows();i++)
    {
        //threshold label
		Index iLabelPredicted=(Index)(std::roundf(mTest(i)));
		iLabelPredicted =std::min<Index>(iLabelPredicted,iNbClass-1);
		iLabelPredicted =std::max<Index>(iLabelPredicted,0);
        _mConfusionMatrix((Index)mRef(i), iLabelPredicted)++;
    }

    //compute accuracy in percent using raw count
    float fTrace= _mConfusionMatrix.trace();
    float fSum= _mConfusionMatrix.sum();
    _accuracy=fTrace/fSum*100.f;

	//compute normalized confusion matrix in percent
    MatrixFloat mSumRow=rowWiseSum(_mConfusionMatrix);
    _mConfusionMatrixNormalized=rowWiseDivide(_mConfusionMatrix,mSumRow)*100.f;

	//compute balaced accuracy
	_balancedAccuracy=_mConfusionMatrixNormalized.trace()/_mConfusionMatrixNormalized.rows();
}
///////////////////////////////////////////////////////////////////////////////