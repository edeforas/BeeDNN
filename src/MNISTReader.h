/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#ifndef MNISTReader_
#define MNISTReader_

#include "Matrix.h"
#include "DataSource.h"

#include <string>
using namespace std;

class MNISTReader: public DataSource
{
public:
	virtual bool load(const string & sName) override;

private:
	bool read_Matrix(string sName, MatrixFloat& m);
	void swap_int(unsigned int &i);
};

#endif
