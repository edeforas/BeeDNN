/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#ifndef KMeans_
#define KMeans_

#include "Matrix.h"

class Loss;

class KMeans
{
public:
	KMeans();
    virtual ~KMeans();
    KMeans& operator=(const KMeans& other);

	void set_sizes(int iInputSize,int iNbRef); //input size; total number of centroids, for now 
	void set_loss(const string&  sLoss);

	void predict_class(const MatrixFloat& mIn, MatrixFloat& mClass) const;
	float compute_dist(const MatrixFloat& m1, const MatrixFloat& m2) const;

	MatrixFloat & ref_vectors();
	MatrixFloat & ref_classes();

private:
	MatrixFloat _mRefVectors;
	MatrixFloat _mRefClasses;

	Loss* _pLoss;

	int _iNbRef;
	int _iInputSize;
};

#endif