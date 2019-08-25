/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#include <algorithm>
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
MatrixFloat colWiseMean(const MatrixFloat& m)
{
#ifdef USE_EIGEN
	return m.colwise().mean();
#else
	int r = m.rows();
	int c = m.cols();
	MatrixFloat result(1,c);

	for (int j = 0; j < c; j++)
	{
		float f = 0.;
		for (int i = 0; i < r; i++)
		{
			f += m(i, j);
		}
		result(0, j) = f / (float)r;
	}

	return result;
#endif
}
///////////////////////////////////////////////////////////////////////////
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
    assert(d.cols() == 1);
    assert(d.rows() == m.rows());

    MatrixFloat r=m;

    for(int l=0;l<r.rows();l++)
        r.row(l)/=d(l);

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

    mPermuted.resizeLike(mIn);

    for (int i = 0; i < mPermutationIndex.rows(); i++)
        mPermuted.row(i) = mIn.row((int)mPermutationIndex(i));
}
///////////////////////////////////////////////////////////////////////////
int argmax(const MatrixFloat& m)
{
    assert(m.rows()==1); //for now, vector raw only

    if(m.cols()==0)
        return 0; //todo error not a vector

    float d=m(0);
    int iIndex=0;

    for(int i=1;i<m.cols();i++)
    {
        if(m(i)>d)
        {
            d=m(i);
            iIndex=i;
        }
    }

    return iIndex;
}
///////////////////////////////////////////////////////////////////////////
void labelToOneHot(const MatrixFloat& mLabel, MatrixFloat& mOneMat, int iNbClass)
{
    assert(mLabel.cols() == 1);

    if (iNbClass == 0)
        iNbClass = (int)mLabel.maxCoeff() + 1; //guess the nb of class

    mOneMat.setZero(mLabel.rows(), iNbClass);

    for (int i = 0; i < mLabel.rows(); i++)
        mOneMat(i, (int)mLabel(i)) = 1;
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
            ss  << m(iL,iR) << " ";
        if(iL+1<m.rows())
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
    vector<float> vf;
    fstream f(sFile,ios::in);
    int iNbLine=0;
    while(!f.eof() && (!f.bad()) && (!f.fail()) )
    {
        string s;
        getline(f,s);

        if(s.empty())
            continue;

        std::replace( s.begin(), s.end(), ',', ' '); //replace ',' by spaces if present

        iNbLine++;

        stringstream ss;
        ss.str(s);
        while(!ss.eof())
        {
            float sF;
            ss >> sF;
            vf.push_back(sF);
        }
    }

    if(iNbLine==0)
        return MatrixFloat();

    MatrixFloat r(iNbLine,(int)vf.size()/iNbLine); // todo check size
    std::copy(vf.begin(),vf.end(),r.data());
    return r;
}
///////////////////////////////////////////////////////////////////////////
const MatrixFloat fromString(const string& s)
{
    MatrixFloat r;
    vector<float> vf;
    stringstream ss(s);
    int iNbCols=0,iNbLine=0;

    while( !ss.eof() )
    {
        float sF;
        ss >> sF;
        vf.push_back(sF);
    }

    iNbCols=(int)std::count(s.begin(),s.end(),'\n')+1;
    iNbLine=(int)vf.size()/iNbCols;

    r.resize(iNbLine,iNbCols);
    std::copy(vf.begin(),vf.end(),r.data());
    return r;
}
///////////////////////////////////////////////////////////////////////////
bool toFile(const string& sFile, const MatrixFloat & m)
{
    fstream f(sFile, ios::out);
    for (int iL = 0; iL < m.rows(); iL++)
    {
        for (int iR = 0; iR < m.cols(); iR++)
            f << m(iL, iR);
        f << endl;
    }

    return true;
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
