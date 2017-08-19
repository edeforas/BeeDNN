#ifndef _MatrixUtil_
#define _MatrixUtil_

#include "Matrix.h"

MatrixFloat rand_perm(unsigned int iSize); //create a vector of index shuffled

MatrixFloat index_to_position(const MatrixFloat& mIndex, unsigned int uiMaxPosition);

MatrixFloat argmax(const MatrixFloat& m);

MatrixFloat decimate(const MatrixFloat& m, unsigned int iRatio);

#endif
