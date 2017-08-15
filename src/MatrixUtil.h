#ifndef _MatrixUtil_
#define _MatrixUtil_

#include "Matrix.h"

Matrix rand_perm(unsigned int iSize); //create a vector of index shuffled

Matrix index_to_position(const Matrix& mIndex, unsigned int uiMaxPosition);

Matrix argmax(const Matrix& m);

#endif
