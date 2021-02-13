/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#ifndef MinMaxScaler_
#define MinMaxScaler_

#include "Matrix.h"

class MinMaxScaler
{
public:
    void fit(const MatrixFloat& m);

    void transform( MatrixFloat& m);


private:
	MatrixFloat _mMin, _mMax;
};

#endif
