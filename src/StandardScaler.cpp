/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#include "StandardScaler.h"

#include <cmath>
#include <cassert>
namespace bee{

///////////////////////////////////////////////////////////////////////////////
void StandardScaler::fit(const MatrixFloat& m)
{
	MatrixFloat mSum = colWiseSum(m);
	MatrixFloat mSumSq = colWiseSumSq(m);

	_mMean.resize(1,m.cols());
	_mStd.resize(1,m.cols());
	
	for (Index i = 0; i < m.cols(); i++)
	{
		float mean = mSum(i) / m.rows();
		float var = mSumSq(i) / m.rows() - (mSum(i)*mSum(i)) / ((float)m.rows()*m.rows()); //todo optimize

		_mMean(i) = mean;

		if (var != 0.f)
			_mStd(i) = sqrtf(var);
		else
			_mStd(i) = 1.f;
	}
}
///////////////////////////////////////////////////////////////////////////////
void StandardScaler::transform( MatrixFloat& m)
{
	for (Index i = 0; i < m.rows(); i++)
		for (Index j = 0; j < m.cols(); j++)
			m(i, j) = (m(i, j) - _mMean(0, j)) / _mStd(0, j);
}
///////////////////////////////////////////////////////////////////////////////
}