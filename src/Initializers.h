/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#ifndef Initializers_
#define Initializers_

#include "Layer.h"
#include "Matrix.h"

// from https://www.tensorflow.org/api_docs/python/tf/keras/initializers


class Initializers
{
public:
	static void GlorotUniform(MatrixFloat &m,Index iInputSize,Index iOutputSize);
    static void GlorotNormal(MatrixFloat& m, Index iInputSize, Index iOutputSize);
    static void HeUniform(MatrixFloat& m, Index iInputSize, Index iOutputSize);
    static void HeNormal(MatrixFloat& m, Index iInputSize, Index iOutputSize);
    static void LecunUniform(MatrixFloat& m, Index iInputSize, Index iOutputSize);
    static void LecunNormal(MatrixFloat& m, Index iInputSize, Index iOutputSize);
    static void Zeros(MatrixFloat &m,Index iInputSize,Index iOutputSize);
    static void Ones(MatrixFloat& m, Index iInputSize, Index iOutputSize);

    static void compute(string sInitializer, MatrixFloat& m, Index iInputSize, Index iOutputSize);
};

#endif
