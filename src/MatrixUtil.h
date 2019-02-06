#ifndef MatrixUtil_
#define MatrixUtil_

#include <string>
using namespace std;

#include "Matrix.h"


MatrixFloat rand_perm(int iSize); //create a vector of index shuffled

MatrixFloat index_to_position(const MatrixFloat& mIndex, int uiMaxPosition);

MatrixFloat argmax(const MatrixFloat& m);

MatrixFloat decimate(const MatrixFloat& m, int iRatio);

namespace MatrixUtil
{
    string to_string(const MatrixFloat& m);
}
#endif
