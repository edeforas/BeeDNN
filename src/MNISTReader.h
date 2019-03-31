/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#ifndef MNISTReader_
#define MNISTReader_

#include "Matrix.h"

#include <string>
using namespace std;

class MNISTReader
{
public:
    bool read_from_folder(const string& sFolder,MatrixFloat& mRefImages,MatrixFloat& mRefLabels,MatrixFloat& mTestImages,MatrixFloat& mTestLabels);
    bool read_Matrix(string sName,MatrixFloat& m);

private:
    void swap_int(unsigned int &i);
};

#endif
