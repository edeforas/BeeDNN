#include <cassert>
#include <cstdlib>
#include <sstream>
#include <iomanip>

#include "MatrixUtil.h"
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
/*MatrixFloat index_to_position(const MatrixFloat& mIndex, int uiMaxPosition)
{
    int iNbRows=mIndex.rows();
    MatrixFloat mPos(iNbRows,uiMaxPosition);
    mPos.setZero();

    for(int i=0;i<iNbRows;i++)
    {
        mPos(i,mIndex(i))=1;
    }

    return mPos;
}
*/
///////////////////////////////////////////////////////////////////////////
int argmax(const MatrixFloat& m)
{
    assert(m.cols()==1); //for now, vector column only

    if(m.cols()==0)
        return 0; //todo error not a vector

    if(m.rows()==0)
        return 0; //todo error empty vector

    float d=m(0,0);
    unsigned int iIndex=0;

    for(int iR=0;iR<m.rows();iR++)
    {
        if(m(iR,0)>d)
        {
            d=m(iR,0);
            iIndex=iR;
        }
    }

    return iIndex;
}
///////////////////////////////////////////////////////////////////////////
MatrixFloat decimate(const MatrixFloat& m, int iRatio)
{
    int iNewSize=m.rows()/iRatio;

    MatrixFloat mDecimated(iNewSize,m.cols());

    for(int i=0;i<iNewSize;i++)
        mDecimated.row(i)=m.row(i*iRatio);

    return mDecimated;
}
///////////////////////////////////////////////////////////////////////////
namespace MatrixUtil
{

string to_string(const MatrixFloat& m)
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
}

///////////////////////////////////////////////////////////////////////////
