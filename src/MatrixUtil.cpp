
#include <cassert>

#include "MatrixUtil.h"


Matrix rand_perm(unsigned int iSize) //create a vector of index shuffled
{
    Matrix m(iSize);

    //create ordered vector
    for(unsigned int i=0;i<iSize;i++)
        m(i)=i;

    //now bubble shuffle
    for(unsigned int i=0;i<iSize;i++)
    {
        unsigned int iNewPos=rand()%iSize;
        double dVal=m(iNewPos);
        m(iNewPos)=m(i);
        m(i)=dVal;
    }

    return m;
}

Matrix index_to_position(const Matrix& mIndex, unsigned int uiMaxPosition)
{
    unsigned int uiNbRows=mIndex.rows();
    Matrix mPos(uiNbRows,uiMaxPosition);
    mPos.set_zero();

    for(unsigned int i=0;i<uiNbRows;i++)
    {
        mPos(i,(unsigned int)mIndex(i))=1;
    }

    return mPos;
}

Matrix argmax(const Matrix& m)
{
    if(m.columns()==0)
        return m;

    Matrix mResult(m.rows(),1);

    for(unsigned int iR=0;iR<m.rows();iR++)
    {
        double d=m(iR,0);
        unsigned int iIndex=0;

        for(unsigned int iC=1;iC<m.columns();iC++)
        {    if(m(iR,iC)>d)
            {
                d=m(iR,iC);
                iIndex=iC;
            }
        }
        mResult(iR)=iIndex;
    }

    return mResult;
}
