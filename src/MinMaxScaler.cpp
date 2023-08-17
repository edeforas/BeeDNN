/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#include "MinMaxScaler.h"

#include <cmath>
#include <cassert>

///////////////////////////////////////////////////////////////////////////////
void MinMaxScaler::fit(const MatrixFloat& m)
{
	_mMin = colWiseMin(m);
	_mMax = colWiseMax(m);
		
	for (Index i = 0; i < m.cols(); i++)
	{ // todo compute directly A and B such as Y=A*x+B
		if (_mMin(i) == _mMax(i))
		{
			_mMin(i) = 0.;
			_mMax(i) = 1.;
		}
	}
}
///////////////////////////////////////////////////////////////////////////////
void MinMaxScaler::transform( MatrixFloat& m)
{
	for (Index i = 0; i < m.rows(); i++)
		for (Index j = 0; j < m.cols(); j++)
			m(i, j) = (m(i, j) - _mMin(j)) / (_mMax(j) - _mMin(j)); //todo optimize
}
///////////////////////////////////////////////////////////////////////////////
