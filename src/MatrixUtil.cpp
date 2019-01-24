#include <cassert>
#include <cstdlib>
#include <sstream>
#include <iomanip>

#include "MatrixUtil.h"
///////////////////////////////////////////////////////////////////////////
MatrixFloat rand_perm(size_t iSize) //create a vector of index shuffled
{
    MatrixFloat m(iSize,1);

    //create ordered vector
    for(size_t i=0;i<iSize;i++)
        m(i,0)=(float)i;

    //now bubble shuffle
    for(size_t i=0;i<iSize;i++)
    {
        unsigned int iNewPos=(size_t)(rand()%iSize);
        float dVal=m(iNewPos,0); //todo, templatize
        m(iNewPos,0)=m(i,0);
        m(i,0)=(float)dVal;
    }

    return m;
}
///////////////////////////////////////////////////////////////////////////
MatrixFloat index_to_position(const MatrixFloat& mIndex, size_t uiMaxPosition)
{
    size_t iNbRows=mIndex.rows();
    MatrixFloat mPos(iNbRows,uiMaxPosition);
    mPos.setZero();

    for(size_t i=0;i<iNbRows;i++)
    {
        mPos(i,(unsigned int)mIndex(i))=1;
    }

    return mPos;
}
///////////////////////////////////////////////////////////////////////////
MatrixFloat argmax(const MatrixFloat& m)
{
    if(m.cols()==0)
        return m;

    MatrixFloat mResult(m.rows(),1);

    for(unsigned int iR=0;iR<m.rows();iR++)
    {
        float d=m(iR,0);
        unsigned int iIndex=0;

        for(unsigned int iC=1;iC<m.cols();iC++)
        {
            if(m(iR,iC)>d)
            {
                d=m(iR,iC);
                iIndex=iC;
            }
        }
        mResult(iR,0)=(float)iIndex;
    }

    return mResult;
}
///////////////////////////////////////////////////////////////////////////
MatrixFloat decimate(const MatrixFloat& m, size_t iRatio)
{
    size_t iNewSize=m.rows()/iRatio;
	
    MatrixFloat mDecimated(iNewSize,m.cols());

    for(size_t i=0;i<iNewSize;i++)
        mDecimated.row(i)=m.row(i*iRatio);
	
	return mDecimated;
}
///////////////////////////////////////////////////////////////////////////
namespace MatrixUtil
{

string to_string(const MatrixFloat& m)
{
    stringstream ss; ss << setprecision(4);
    for(unsigned int iL=0;iL<m.rows();iL++)
    {
        for(unsigned int iR=0;iR<m.cols();iR++)
            ss << setw(10) << m(iL,iR);
        ss << endl;
    }

    return ss.str();
}
}

///////////////////////////////////////////////////////////////////////////
