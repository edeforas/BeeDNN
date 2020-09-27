/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#include "KMeans.h"

#include "Matrix.h"

#include <cmath>
using namespace std;

/////////////////////////////////////////////////////////////////////////////////////////////////
KMeans::KMeans()
{ 
	_iNbRef = 0;
	_iInputSize = 0;
}
/////////////////////////////////////////////////////////////////////////////////////////////////
KMeans::~KMeans()
{ }
/////////////////////////////////////////////////////////////////////////////////////////////////
/*Net& Net::operator=(const Net& other)
{
    clear();

    for(size_t i=0;i<other._layers.size();i++)
        _layers.push_back(other._layers[i]->clone());

    _bClassificationMode = other._bClassificationMode;

    return *this;
}
*/
/////////////////////////////////////////////////////////////////////////////////////////////////
void KMeans::set_sizes(int iInputSize, int iNbRef) //input size; total number of centroids, for now 
{
	_iInputSize = iInputSize;
	_iNbRef = iNbRef;

	_mRefVectors.resize(_iNbRef, _iInputSize);
	_mRefClasses.resize(_iNbRef, 1);
}
/////////////////////////////////////////////////////////////////////////////////////////////
MatrixFloat & KMeans::ref_vectors()
{
	return _mRefVectors;
}
/////////////////////////////////////////////////////////////////////////////////////////////
MatrixFloat & KMeans::ref_classes()
{
	return _mRefClasses;
}
/////////////////////////////////////////////////////////////////////////////////////////////
void KMeans::predict(const MatrixFloat& mIn, MatrixFloat& mClass) const
{
    MatrixFloat mOut;

	mClass.resize(mIn.rows(), 1);

	if (_mRefVectors.size() == 0)
	{
		mClass.setZero();
		return;
	}

	for (int i = 0; i < mIn.rows(); i++)
	{
		float fDistBest = 1.e38f; //todo
		int indexBest = -1;
		for (int j = 0; j < _mRefVectors.rows(); j++)
		{
			float d = compute_dist(mIn.row(i), _mRefVectors.row(j));
			if (d < fDistBest)
			{
				fDistBest = d;
				indexBest = j;
			}
		}
	
		assert(indexBest != -1);
		mClass(i) = _mRefClasses(indexBest);
	}
}
/////////////////////////////////////////////////////////////////////////////////////////////
// todo use loss instead
float KMeans::compute_dist(const MatrixFloat& m1, const MatrixFloat& m2) const
{
	assert(m1.rows() == 1);
	assert(m2.rows() == 1);

	assert(m1.cols() == m2.cols());
	return (m1 - m2).squaredNorm();
}
/////////////////////////////////////////////////////////////////////////////////////////////
