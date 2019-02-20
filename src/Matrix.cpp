#include "Matrix.h"

//matrix view on another matrix, without malloc and copy
const MatrixFloat from_raw_buffer(float *pBuffer,int iRows,int iCols)
{
#ifdef USE_EIGEN
    return Eigen::Map<MatrixFloat>((float*)pBuffer,static_cast<Eigen::Index>(iRows),static_cast<Eigen::Index>(iCols));
#else
    return MatrixFloat(pBuffer,iRows,iCols);
#endif
}


