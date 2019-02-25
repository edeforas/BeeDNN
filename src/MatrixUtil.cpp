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
///////////////////////////////////////////////////////////////////////////
void kronecker(int i,int iSize,MatrixFloat& m)
{
    m.resize(iSize,1);
    m.setZero();
    m(i,0)=1;
}
///////////////////////////////////////////////////////////////////////////

}
