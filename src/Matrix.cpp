#include <cassert>
#include <cstdlib>
#include <sstream>
#include <iomanip>

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
    MatrixFloat result(r,1);

    for(int i=0;i<r;i++)
        result(i,0)=(m.row(i)).sum();

    return result;
#endif
}
///////////////////////////////////////////////////////////////////////////
MatrixFloat rand_perm(int iSize) //create a vector of index shuffled
{
    MatrixFloat m(iSize,1);

    //create ordered vector
    for(int i=0;i<iSize;i++)
        m(i,0)=(float)i;

    //now bubble shuffle
    for(int i=0;i<iSize;i++)
    {
        int iNewPos=(int)(rand()%iSize);
        float dVal=m(iNewPos,0); //todo, templatize
        m(iNewPos,0)=m(i,0);
        m(i,0)=(float)dVal;
    }

    return m;
}
///////////////////////////////////////////////////////////////////////////
int argmax(const MatrixFloat& m)
{
    assert(m.rows()==1); //for now, vector raw only

    if(m.cols()==0)
        return 0; //todo error not a vector

    float d=m(0,0);
    unsigned int iIndex=0;

    for(int i=0;i<m.cols();i++)
    {
        if(m(0,i)>d)
        {
            d=m(0,i);
            iIndex=i;
        }
    }

    return iIndex;
}
///////////////////////////////////////////////////////////////////////////
MatrixFloat decimate(const MatrixFloat& m, int iRatio)
{
    int iNewSize=(int)(m.rows()/iRatio);

    MatrixFloat mDecimated(iNewSize,m.cols());

    for(int i=0;i<iNewSize;i++)
        mDecimated.row(i)=m.row(i*iRatio);

    return mDecimated;
}
///////////////////////////////////////////////////////////////////////////
string matrix_to_string(const MatrixFloat& m)
{
    stringstream ss; ss << setprecision(4);
    for(int iL=0;iL<m.rows();iL++)
    {
        for(int iR=0;iR<m.cols();iR++)
            ss << setw(10) << m(iL,iR);
        ss << endl;
    }

    return ss.str();
}
///////////////////////////////////////////////////////////////////////////
