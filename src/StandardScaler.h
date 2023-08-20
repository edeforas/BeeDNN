/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#ifndef StandardScaler_
#define StandardScaler_

#include "Matrix.h"
namespace bee {
class StandardScaler
{
public:
    void fit(const MatrixFloat& m);

    void transform( MatrixFloat& m);


private:
	MatrixFloat _mMean, _mStd;
};
}
#endif
