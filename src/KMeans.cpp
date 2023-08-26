/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#include "KMeans.h"

#include "Matrix.h"

#include <cmath>
#include "Loss.h"

using namespace std;
namespace beednn {

/////////////////////////////////////////////////////////////////////////////////////////////////
KMeans::KMeans()
{ 
	_iNbRef = 0;
	_iInputSize = 0;
	_pLoss = create_loss("MeanSquaredError");
}
/////////////////////////////////////////////////////////////////////////////////////////////////
KMeans::~KMeans()
{ 
	delete _pLoss;
}
/////////////////////////////////////////////////////////////////////////////////////////////////
KMeans& KMeans::operator=(const KMeans& other)
{
	_mRefVectors=other._mRefVectors;
	_mRefClasses=other._mRefClasses;

	_pLoss=create_loss(other._pLoss->name());

	_iNbRef=other._iNbRef;
	_iInputSize = other._iInputSize;

    return *this;
}
/////////////////////////////////////////////////////////////////////////////////////////////////
void KMeans::set_sizes(int iInputSize, int iNbRef) //input size; total number of centroids, for now 
{
	_iInputSize = iInputSize;
	_iNbRef = iNbRef;

	_mRefVectors.resize(_iNbRef, _iInputSize);
	_mRefClasses.resize(_iNbRef, 1);
}
/////////////////////////////////////////////////////////////////////////////////////////////
void KMeans::set_loss(const string&  sLoss)
{
	delete _pLoss;
	_pLoss = create_loss(sLoss);
	assert(_pLoss);
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
void KMeans::predict_classes(const MatrixFloat& mIn, MatrixFloat& mClass) const
{
    MatrixFloat mOut;

	mClass.resize(mIn.rows(), 1);

	if (_mRefVectors.size() == 0)
	{
		mClass.setZero();
		return;
	}

	for (int iS = 0; iS < mIn.rows(); iS++)
	{
		float fDistBest = 1.e38f; //todo
		int indexBest = -1;
		MatrixFloat mS = mIn.row(iS);

		for (int iR = 0; iR < _mRefVectors.rows(); iR++)
		{
			float d = compute_dist(mS, _mRefVectors.row(iR));
			if (d < fDistBest)
			{
				fDistBest = d;
				indexBest = iR;
			}
		}
	
		assert(indexBest != -1);
		mClass(iS) = _mRefClasses(indexBest);
	}
}
/////////////////////////////////////////////////////////////////////////////////////////////
float KMeans::compute_dist(const MatrixFloat& m1, const MatrixFloat& m2) const
{
	assert(m1.rows() == 1);
	assert(m2.rows() == 1);
	assert(m1.cols() == m2.cols());

	MatrixFloat mLoss;
	_pLoss->compute(m1, m2, mLoss);
	return mLoss.mean();
}
/////////////////////////////////////////////////////////////////////////////////////////////
}