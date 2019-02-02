#ifndef MatrixUtil_
#define MatrixUtil_

#include <string>
using namespace std;

#include "Matrix.h"



MatrixFloat rand_perm(size_t iSize); //create a vector of index shuffled

MatrixFloat index_to_position(const MatrixFloat& mIndex, size_t uiMaxPosition);

MatrixFloat argmax(const MatrixFloat& m);

MatrixFloat decimate(const MatrixFloat& m, size_t iRatio);

namespace MatrixUtil
{
    string to_string(const MatrixFloat& m);
}
#endif
