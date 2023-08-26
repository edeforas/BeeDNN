/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#pragma once

#include "Matrix.h"
#include "DataSource.h"

#include <string>

namespace beednn {
class MNISTReader: public DataSource
{
public:
	virtual bool load(const std::string & sName) override;

private:
	bool read_Matrix(std::string sName, MatrixFloat& m);
	void swap_int(unsigned int &i);
};
}
