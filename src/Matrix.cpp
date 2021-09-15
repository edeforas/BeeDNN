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
const MatrixFloatView fromRawBuffer(const float *pBuffer,Index iRows,Index iCols)
{
#ifdef USE_EIGEN
    return Eigen::Map<MatrixFloat>((float*)pBuffer,static_cast<Eigen::Index>(iRows),static_cast<Eigen::Index>(iCols));
#else
    return MatrixFloat((float*)pBuffer,iRows,iCols);
#endif
}
///////////////////////////////////////////////////////////////////////////
const MatrixFloatView viewResize(const MatrixFloat& m, Index iRows, Index iCols)
{
	assert(m.size() == iRows * iCols);

	return fromRawBuffer(m.data(), iRows, iCols);
}
///////////////////////////////////////////////////////////////////////////
MatrixFloatView fromRawBuffer(float *pBuffer,Index iRows,Index iCols)
{
#ifdef USE_EIGEN
    return Eigen::Map<MatrixFloat>(pBuffer,static_cast<Eigen::Index>(iRows),static_cast<Eigen::Index>(iCols));
#else
    return MatrixFloat(pBuffer,iRows,iCols);
#endif
}
///////////////////////////////////////////////////////////////////////////
MatrixFloatView createView(MatrixFloat & mRef)
{
	return fromRawBuffer(mRef.data(), mRef.rows(), mRef.cols());
}
///////////////////////////////////////////////////////////////////////////
void copyInto(const MatrixFloat& mToCopy, MatrixFloat& m, Index iStartRow)
{
#ifdef USE_EIGEN
	m.block(iStartRow, 0, mToCopy.rows(), mToCopy.cols()) = mToCopy;
#else
	std::copy(mToCopy.data(), mToCopy.data() + mToCopy.size(), m.data()+ iStartRow*m.cols());
#endif
}
///////////////////////////////////////////////////////////////////////////
MatrixFloat rowWiseSum(const MatrixFloat& m)
{
#ifdef USE_EIGEN
	return m.rowwise().sum();
#else
	Index r = m.rows();
	Index c = m.cols();
	MatrixFloat result(r,1);

	for (Index i = 0; i < r; i++)
	{
		float sum = 0.f;
		for (Index j = 0; j < c; j++)
			sum += m(i, j);

		result(i) = sum;
	}
	return result;
#endif
}
///////////////////////////////////////////////////////////////////////////
MatrixFloat rowWiseMean(const MatrixFloat& m)
{
#ifdef USE_EIGEN
	return m.rowwise().mean();
#else
	Index r = m.rows();
	Index c = m.cols();
	MatrixFloat result(r, 1);

	for (Index i = 0; i < r; i++)
	{
		float sum = 0.f;
		for (Index j = 0; j < c; j++)
			sum += m(i, j);

		result(i) = sum/c;
	}
	return result;
#endif
}
///////////////////////////////////////////////////////////////////////////
MatrixFloat rowWiseSumSq(const MatrixFloat& m)
{
#ifdef USE_EIGEN
	return m.rowwise().sum();
#else
	Index r = m.rows();
	Index c = m.cols();
	MatrixFloat result(r, 1);

	for (Index i = 0; i < r; i++)
	{
		float sum = 0.f;
		for (Index j = 0; j < c; j++)
			sum += m(i, j)* m(i, j);

		result(i) = sum;
	}
	return result;
#endif
}
///////////////////////////////////////////////////////////////////////////
MatrixFloat colWiseSum(const MatrixFloat& m)
{
#ifdef USE_EIGEN
	return m.colwise().sum();
#else
	Index r = m.rows();
	Index c = m.cols();
	MatrixFloat result(1,c);

	for (Index j = 0; j < c; j++)
	{
		float sum = 0.f;
		for (Index i = 0; i < r; i++)
			sum += m(i, j);

		result(j) = sum;
	}
	return result;
#endif
}
///////////////////////////////////////////////////////////////////////////
MatrixFloat colWiseSumSq(const MatrixFloat& m)
{
#ifdef USE_EIGEN
	return (m.array().square()).colwise().sum();
#else
	Index r = m.rows();
	Index c = m.cols();
	MatrixFloat result(1,c);

	for (Index j = 0; j < c; j++)
	{
		float sum = 0.f;
		for (Index i = 0; i < r; i++)
			sum += m(i, j) * m(i, j);

		result(j) = sum;
	}
	return result;
#endif
}
///////////////////////////////////////////////////////////////////////////
MatrixFloat colWiseMean(const MatrixFloat& m)
{
#ifdef USE_EIGEN
	return m.colwise().mean();
#else
	Index r = m.rows();
	Index c = m.cols();
	MatrixFloat result(1,c);

	for (Index j = 0; j < c; j++)
	{
		float sum = 0.f;
		for (Index i = 0; i < r; i++)
			sum += m(i, j);

		result(j) = sum/r;
	}
	return result;
#endif
}
///////////////////////////////////////////////////////////////////////////
MatrixFloat colWiseMin(const MatrixFloat& m)
{
#ifdef USE_EIGEN
	return m.colwise().minCoeff();
#else
	Index r = m.rows();
	Index c = m.cols();
	MatrixFloat result(1, c);

	for (Index j = 0; j < c; j++)
	{
		float fMin = m(0, j); // todo test empty matrix
		for (Index i = 0; i < r; i++)
			fMin =std::min(fMin, m(i, j));

		result(j) = fMin;
	}
	return result;
#endif
}
///////////////////////////////////////////////////////////////////////////
MatrixFloat colWiseMax(const MatrixFloat& m)
{
#ifdef USE_EIGEN
	return m.colwise().maxCoeff();
#else
	Index r = m.rows();
	Index c = m.cols();
	MatrixFloat result(1, c);

	for (Index j = 0; j < c; j++)
	{
		float fMax = m(0, j); // todo test empty matrix
		for (Index i = 0; i < r; i++)
			fMax = std::max(fMax, m(i, j));

		result(j) = fMax;
	}
	return result;
#endif
}
///////////////////////////////////////////////////////////////////////////
MatrixFloat rowWiseMult(const MatrixFloat& m, const MatrixFloat& d)
{
    assert(d.rows() == m.rows());
	assert(d.cols() == 1);

    MatrixFloat r=m;

    for(Index l=0;l<r.rows();l++)
        r.row(l)*=d(l);

    return r;
}
///////////////////////////////////////////////////////////////////////////
MatrixFloat rowWiseDivide(const MatrixFloat& m, const MatrixFloat& d)
{
    assert(d.cols() == 1);
    assert(d.rows() == m.rows());

    MatrixFloat r=m;

    for(Index l=0;l<r.rows();l++)
        r.row(l)/=d(l);

    return r;
}
///////////////////////////////////////////////////////////////////////////
MatrixFloat rowWiseAdd(const MatrixFloat& m, const MatrixFloat& d)
{
    assert(d.rows() == 1);
    assert(d.cols() == m.cols());

	MatrixFloat r = m;

//#ifdef USE_EIGEN
//	r.rowwise() += d;// error cannot use a vector
//#else
    for (Index l = 0; l < r.rows(); l++)
        r.row(l) += d;
//#endif
	return r;
}
///////////////////////////////////////////////////////////////////////////
vector<Index> randPerm(Index iSize) //create a vector of index shuffled
{
	vector<Index> v(iSize);
	for (Index i = 0; i < iSize; i++)
		v[i] = i;

	std::shuffle(v.begin(), v.end(), randomEngine());

    return v;
}
///////////////////////////////////////////////////////////////////////////
void applyRowPermutation(const vector<Index>& vPermutation, const MatrixFloat & mIn, MatrixFloat & mPermuted)
{
    assert((Index)vPermutation.size() == mIn.rows());

    mPermuted.resizeLike(mIn);

	for (Index i = 0; i < (Index)(vPermutation.size()); i++)
	{
		Index iPerm= vPermutation[i];

		assert(iPerm >=0);
		assert(iPerm < mIn.rows());

		mPermuted.row(i) = mIn.row(iPerm);
	}
}
///////////////////////////////////////////////////////////////////////////
void clamp(MatrixFloat& m, float fClampMin, float fClampMax)
{
	for (Index i = 0; i < m.size(); i++)
	{
		if (m(i) > fClampMax)
			m(i) = fClampMax;
		else if (m(i) < fClampMin)
			m(i) = fClampMin;
	}
}
///////////////////////////////////////////////////////////////////////////
Index argmax(const MatrixFloat& m)
{
    assert(m.rows()==1); //for now, vector raw only

    if(m.cols()==0)
        return 0; //todo error not a vector

    float d=m(0);
    Index iIndex=0;

    for(Index i=1;i<m.cols();i++)
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
void rowsArgmax(const MatrixFloat& m, MatrixFloat& argM)
{
    Index iRows = (Index)m.rows();
    argM.resize(iRows, 1);

    for (Index i = 0; i < iRows; i++)
        argM(i) = (float)argmax(m.row(i));
}
///////////////////////////////////////////////////////////////////////////
MatrixFloat decimate(const MatrixFloat& m, Index iRatio)
{
    Index iNewSize=(Index)(m.rows()/iRatio);

    MatrixFloat mDecimated(iNewSize,m.cols());

    for(Index i=0;i<iNewSize;i++)
        mDecimated.row(i)=m.row(i*iRatio);

    return mDecimated;
}
///////////////////////////////////////////////////////////////////////////
string toString(const MatrixFloat& m)
{
    stringstream ss; ss << setprecision(4);
    for(Index iL=0;iL<m.rows();iL++)
    {
        for(Index iR=0;iR<m.cols();iR++)
            ss  << m(iL,iR) << " ";
        if(iL+1<m.rows())
            ss << endl;
    }

    return ss.str();
}
///////////////////////////////////////////////////////////////////////////
const MatrixFloat fromFile(const string& sFile)
{    
    vector<float> vf;
    fstream f(sFile,ios::in);
    Index iNbLine=0;
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

    MatrixFloat r(iNbLine,(Index)vf.size()/iNbLine); // todo check size
    std::copy(vf.begin(),vf.end(),r.data());
    return r;
}
///////////////////////////////////////////////////////////////////////////
const MatrixFloat fromString(const string& s)
{
    MatrixFloat r;
    vector<float> vf;
    stringstream ss(s);
    Index iNbCols=0,iNbLine=0;

    while( !ss.eof() )
    {
        float sF;
        ss >> sF;
        vf.push_back(sF);
    }

    iNbCols=(Index)std::count(s.begin(),s.end(),'\n')+1;
    iNbLine=(Index)vf.size()/iNbCols;

    r.resize(iNbLine,iNbCols);
    std::copy(vf.begin(),vf.end(),r.data());
    return r;
}
///////////////////////////////////////////////////////////////////////////
bool toFile(const string& sFile, const MatrixFloat & m)
{
    fstream f(sFile, ios::out);
    for (Index iL = 0; iL < m.rows(); iL++)
    {
        for (Index iR = 0; iR < m.cols(); iR++)
            f << m(iL, iR) << " ";
        f << endl;
    }

    return true;
}
///////////////////////////////////////////////////////////////////////////
//create a row view starting at iStartRow ending at iEndRow (not included)
const MatrixFloat viewRow(const MatrixFloat& m, Index iStartRow, Index iEndRow)
{
    assert(iStartRow < iEndRow); //iEndRow not included
    assert(m.rows() >= iEndRow);

    return fromRawBuffer(m.data() + iStartRow * m.cols(), iEndRow- iStartRow, (Index)m.cols());
}
///////////////////////////////////////////////////////////////////////////
default_random_engine& randomEngine()
{
	static default_random_engine rng;
	return rng;
}
///////////////////////////////////////////////////////////////////////////
void setRandomUniform(MatrixFloat& m, float fMin, float fMax)
{
	uniform_real_distribution<float> dis(fMin, fMax);

	for (Index i = 0; i < m.size(); i++)
		m(i) = dis(randomEngine());
}
///////////////////////////////////////////////////////////////////////////
void setQuickBernoulli(MatrixFloat& m, float fProba)
{
	// quick bernoulli ; resolution proba = 1/65536.
	// speed is 6x faster than bernoulli_distribution !
	unsigned int uiLimit = int(fProba*65536.);
	for (Index i = 0; i < m.size(); i++)
		m(i) = (randomEngine()() & 0xffff) < uiLimit; //quick, precise enough

/*
	bernoulli_distribution dis(fProba);
	for (Index i = 0; i < m.size(); i++)
		m(i) = (float)(dis(randomEngine())); //slow 
*/
}
///////////////////////////////////////////////////////////////////////////
void channelWiseAdd(MatrixFloat& mIn, Index iNbSamples, Index iNbChannels, Index iNbRows, Index iNbCols, const MatrixFloat& weight)
{
	assert(weight.size() == iNbChannels);
	assert(mIn.rows() == iNbSamples);
	assert(mIn.size() == iNbSamples * iNbChannels*iNbRows*iNbCols);

	//todo optimize a lot
	for (Index iS = 0; iS < iNbSamples; iS++)
		for (Index iH = 0; iH < iNbChannels; iH++)
			for (Index iR = 0; iR < iNbRows; iR++)
				for (Index iC = 0; iC < iNbCols; iC++)
					mIn(iS, iH*iNbRows*iNbCols + iR * iNbCols + iC) += weight(iH);
}
///////////////////////////////////////////////////////////////////////////
MatrixFloat channelWiseMean(const MatrixFloat& m, Index iNbSamples, Index iNbChannels, Index iNbRows, Index iNbCols)
{
	assert(m.rows() == iNbSamples);
	assert(m.size() == iNbSamples * iNbChannels*iNbRows*iNbCols);

	MatrixFloat mMean;
	mMean.setZero(1, iNbChannels);

	//todo optimize a lot
	for (Index iS = 0; iS < iNbSamples; iS++)
		for (Index iH = 0; iH < iNbChannels; iH++)
			for (Index iR = 0; iR < iNbRows; iR++)
				for (Index iC = 0; iC < iNbCols; iC++)
					mMean(0, iH) += m(iS, iH*iNbRows*iNbCols + iR * iNbCols + iC);
					
	mMean *= (1.f / iNbSamples * iNbRows*iNbCols);

	return mMean;
}
///////////////////////////////////////////////////////////////////////////
float * rowPtr(MatrixFloat& m, Index iRow)
{
	return m.data() + m.cols()*iRow;
}
///////////////////////////////////////////////////////////////////////////
const float * rowPtr(const MatrixFloat& m, Index iRow)
{
	return m.data() + m.cols()*iRow;
}
///////////////////////////////////////////////////////////////////////////
MatrixFloat tanh(const MatrixFloat& m)
{
	MatrixFloat r = m;
	for (Index i = 0; i < r.size(); i++)
		r(i)=::tanh(m(i));

	return r;
}
///////////////////////////////////////////////////////////////////////////
void reverseData(float* pData, Index iSize)
{
	float* pDataEnd = pData + iSize - 1;
	Index iHalfSize = iSize >> 1;
	for (Index i = 0; i <= iHalfSize; i++)
		*pData++ = *pDataEnd--;
}
///////////////////////////////////////////////////////////////////////////
