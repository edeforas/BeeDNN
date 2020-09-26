/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#ifndef KMeans_
#define KMeans_

#include "Matrix.h"

class KMeans
{
public:
	KMeans();
    virtual ~KMeans();
    //Net& operator=(const Net& other);

	void set_sizes(int iInputSize,int iNbRef); //input size; total number of centroids, for now 

	void classify(const MatrixFloat& mIn, MatrixFloat& mClass) const;

	MatrixFloat & ref_vectors();
	MatrixFloat & ref_classes();

private:
	float compute_dist(const MatrixFloat& m1, const MatrixFloat& m2) const;

	MatrixFloat _mRefVectors;
	MatrixFloat _mRefClasses;

	int _iNbRef;
	int _iInputSize;
};

#endif