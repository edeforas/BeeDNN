/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#ifndef KMeans_
#define KMeans_

#include <vector>
using namespace std;

class Layer;
#include "Matrix.h"

class KMeans
{
public:
	KMeans();
    virtual ~KMeans();
    //Net& operator=(const Net& other);

	void clear();
	void init();

	void set_ref_by_classes(int iRefByClasses);

	void classify(const MatrixFloat& mIn, MatrixFloat& mClass) const;

    void set_train_mode(bool bTrainMode); // set to true if training, set to false if testing (default)

	MatrixFloat _mRefVectors;
	MatrixFloat _mRefClasses;

private:
	int _iRefByCLasses;
	bool _bTrainMode;
	bool _bClassificationMode;
};

#endif
