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

MatrixFloat rowWiseSum(const MatrixFloat& m)
{
#ifdef USE_EIGEN
    return m.rowwise().sum();
#else
    int r=m.rows();
    MatrixFloat rs(r,1);

    for(int i=0;i<r;i++)
        rs(i,0)=(rs.row(i)).sum();

    return rs;
#endif
}
