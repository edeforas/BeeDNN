/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#include <cstdlib>
#include <sstream>
#include <fstream>
#include <random>
#include <vector>
#include <iomanip>

#include "Matrix.h"
///////////////////////////////////////////////////////////////////////////
//matrix view on another matrix, without malloc and copy
const MatrixFloatView fromRawBuffer(const float *pBuffer,int iRows,int iCols)
{
#ifdef USE_EIGEN
    return Eigen::Map<MatrixFloat>((float*)pBuffer,static_cast<Eigen::Index>(iRows),static_cast<Eigen::Index>(iCols));
#else
    return MatrixFloat((float*)pBuffer,iRows,iCols);
#endif
}
///////////////////////////////////////////////////////////////////////////
MatrixFloatView fromRawBuffer(float *pBuffer,int iRows,int iCols)
{
#ifdef USE_EIGEN
    return Eigen::Map<MatrixFloat>(pBuffer,static_cast<Eigen::Index>(iRows),static_cast<Eigen::Index>(iCols));
#else
    return MatrixFloat(pBuffer,iRows,iCols);
#endif
}
///////////////////////////////////////////////////////////////////////////
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
MatrixFloat cwiseLog(const MatrixFloat& m)
{
#ifdef USE_EIGEN
	return m.array().log();
#else
	MatrixFloat result(m.rows(),m.cols());

	for (int i = 0; i < m.size(); i++)
		result(i) = log(m(i));

	return result;
#endif
}
///////////////////////////////////////////////////////////////////////////
MatrixFloat cwiseExp(const MatrixFloat& m)
{
#ifdef USE_EIGEN
    return m.array().exp();
#else
    MatrixFloat result(m.rows(),m.cols());

    for (int i = 0; i < m.size(); i++)
        result(i) = exp(m(i));

    return result;
#endif
}///////////////////////////////////////////////////////////////////////////
void arraySub(MatrixFloat& m,float f)
{
#ifdef USE_EIGEN
    m.array()-=f;
#else
    m-=f;
#endif
}
///////////////////////////////////////////////////////////////////////////
MatrixFloat rowWiseDivide(const MatrixFloat& m, const MatrixFloat& d)
{
	assert(d.rows() == 1);
	assert(d.cols() == m.cols());

	MatrixFloat r=m;

    for(int l=0;l<r.rows();l++)
        r.row(l)/=d(l,0);

    return r;
}
///////////////////////////////////////////////////////////////////////////
MatrixFloat rowWiseAdd(const MatrixFloat& m, const MatrixFloat& d)
{
	assert(d.rows() == 1);
	assert(d.cols() == m.cols());

#ifdef USE_EIGEN
    return m + d.replicate(m.rows(), 1);
#else
	MatrixFloat r = m;
	for (int l = 0; l < r.rows(); l++)
		r.row(l) += d;
	return r;
#endif
}
///////////////////////////////////////////////////////////////////////////
MatrixFloat randPerm(int iSize) //create a vector of index shuffled
{
    MatrixFloat m(iSize,1);

    //create ordered vector
    for(int i=0;i<iSize;i++)
        m(i,0)=(float)i;

    std::shuffle(m.data(),m.data()+m.size(), std::default_random_engine());

    return m;
}
///////////////////////////////////////////////////////////////////////////
void applyRowPermutation(const MatrixFloat & mPermutationIndex, const MatrixFloat & mIn, MatrixFloat & mPermuted)
{
	assert(mPermutationIndex.rows() == mIn.rows());
	assert(mPermutationIndex.cols() == 1);

	mPermuted.resize(mIn.rows(), mIn.cols());

	for (int i = 0; i < mPermutationIndex.rows(); i++)
		mPermuted.row(i) = mIn.row((int)mPermutationIndex(i));
}
///////////////////////////////////////////////////////////////////////////
int argmax(const MatrixFloat& m)
{
    assert(m.rows()==1); //for now, vector raw only

    if(m.cols()==0)
        return 0; //todo error not a vector

    float d=m(0,0);
    int iIndex=0;

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
void rowsArgmax(const MatrixFloat& m, MatrixFloat& argM)
{
    int iRows = (int)m.rows();
	argM.resize(iRows, 1);

	for (int i = 0; i < iRows; i++)
        argM(i) = (float)argmax(m.row(i));
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
string toString(const MatrixFloat& m)
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
void contatenateVerticallyInto(const MatrixFloat& mA, const MatrixFloat& mB, MatrixFloat& mAB)
{
    assert(mA.cols()== mB.cols());

    int iRowA = (int)mA.rows();
    int iRowB = (int)mB.rows();
    int iCols = (int)mA.cols();

    mAB.resize(iRowA + iRowB, iCols);

#ifdef USE_EIGEN
    mAB << mA , mB;
#else
    //todo check mA and mB are not view on other matrixes with reduced columns (horizontal stride pb)
    std::copy(mA.data(), mA.data() + mA.size(), mAB.data());
    std::copy(mB.data(), mB.data() + mB.size(), mAB.data() + mA.size());
#endif
}
///////////////////////////////////////////////////////////////////////////
const MatrixFloat addColumnOfOne(const MatrixFloat& m)
{
    // todo : slow
    MatrixFloat r(m.rows(), m.cols() + 1);

    for (int iL = 0; iL < m.rows(); iL++)
    {
        for (int iR = 0; iR < m.cols(); iR++)
            r(iL,iR)= m(iL, iR);
        r(iL, m.cols()) = 1.f;
    }

    return r;
}
///////////////////////////////////////////////////////////////////////////
const MatrixFloat fromFile(const string& sFile)
{
    MatrixFloat r;
    vector<float> vf;
    fstream f(sFile,ios::in);
    int iNbCols=0,iNbLine=0;
    while(!f.eof() && (!f.bad()) && (!f.fail()) )
    {
        string s;
        getline(f,s);
        iNbLine++;
        if(iNbCols==0)
        {
            //count nb of columns
            int iNbSpace=(int)std::count(s.begin(),s.end(),' ');
            iNbCols=iNbSpace+1;
        }

        stringstream ss;
        ss.str(s);
        for(int i=0;i<iNbCols;i++)
        {
            float sF;
            ss >> sF;
            vf.push_back(sF);
        }

        r.resize(iNbLine,iNbCols);
        std::copy(vf.begin(),vf.end(),r.data());
    }

    return r;
}
///////////////////////////////////////////////////////////////////////////
 //create a row view starting at iStartRow ending at iEndRow (not included)
const MatrixFloat rowRange(const MatrixFloat& m, int iStartRow, int iEndRow)
{
	assert(iStartRow < iEndRow); //iEndRow not included
	assert(m.rows() >= iEndRow);

	return fromRawBuffer(m.data() + iStartRow * m.cols(), iEndRow- iStartRow, (int)m.cols());
}
///////////////////////////////////////////////////////////////////////////
