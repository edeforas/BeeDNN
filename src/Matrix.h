/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#ifndef Matrix_
#define Matrix_

#include <algorithm>
#include <cassert>
#include <string>
#include <vector>
#include <cmath>
#include <random>

#ifdef USE_EIGEN

//#define EIGEN_DONT_PARALLELIZE // keep the cpu core for upper algorithms
#include <Eigen/Core>

namespace bee {

using Index=Eigen::Index;
using string=std::string;
template<class T> using vector=std::vector<T>;
//using namespace Eigen;
typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixFloat;
typedef Eigen::Map<MatrixFloat> MatrixFloatView;

#else

typedef ptrdiff_t Eigen::Index;

template <class T>
class Matrix
{
public:
    Matrix<T>()
    {
        _iRows=0;
        _iColumns=0;
        _iSize=_iRows*_iColumns;
        _data=0;
        _bIsView=false;
    }

    Matrix<T>(Eigen::Index iRows, Eigen::Index iColumns)
    {
        _iRows=iRows;
        _iColumns=iColumns;
        _iSize=_iRows*_iColumns;
        _data=new T[_iSize];
        _bIsView=false;
    }
    
    Matrix<T>(T* pData,Eigen::Index iRows,Eigen::Index iColumns)
    {
        _iRows=iRows;
        _iColumns=iColumns;
        _iSize=_iRows*_iColumns;
        _data=pData;
        _bIsView=true;
    }
    
    static const Matrix<T> from_raw_buffer(const T* pData,Eigen::Index iRows,Eigen::Index iColumns)
    {
        Matrix<T> m;

        m._iRows=iRows;
        m._iColumns=iColumns;
        m._iSize=iRows*iColumns;
        m._data=(T*)pData;
        m._bIsView=true;

        return m;
    }

    Matrix<T>(const Matrix<T> &a)
    {
        _iRows=a._iRows;
        _iColumns=a._iColumns;
        _iSize=_iRows*_iColumns;
        _data=new T[_iSize];
        _bIsView=false;

        for( Eigen::Index i=0;i< _iSize;i++)
            _data[i]=a(i);
        //todo use or merge with operator=()(a); ??
    }

    ~Matrix<T>()
    {
        if(!_bIsView)
            delete [] _data;
    }

    void assign(T* first,T* last)
    {
        resize(1,(Eigen::Index)(last-first));
        for(Eigen::Index i=0;i<size();i++)
            operator()(i)=*first++;

        //todo  check and optimize
    }
    
    Matrix<T>& operator=( const Matrix<T>& b)
    {
        resize(b.rows(),b.cols());
        
        for(Eigen::Index i=0;i<size();i++)
            operator()(i)=b(i);
        
        return *this;
    }
    
	Matrix<T>& array() //for eigen compatibility
	{
		return *this;
	}
	const Matrix<T>& array() const //for eigen compatibility
	{
		return *this;
	}
	
	Matrix<T> operator-() const
	{
		Matrix<T> m(_iRows,_iColumns);
		for (Eigen::Index i = 0; i < size(); i++)
			m._data[i] = -_data[i];

		return m;
	}

	Eigen::Index rows() const
    {
        return _iRows;
    }

	Eigen::Index cols() const
    {
        return _iColumns;
    }
    
	Eigen::Index size() const
    {
        return _iSize;
    }

    void resize(Eigen::Index iRows, Eigen::Index iColumns) // slow function!
    {
        if((iColumns==_iColumns) && ( iRows==_iRows))
            return;

        _iRows=iRows;
        _iColumns=iColumns;
        Eigen::Index iSize=_iRows*_iColumns;
		if (iSize == _iSize)
			return;

		_iSize = iSize;

        if(!_bIsView)
            delete[] _data;
        else
            _bIsView=false;

        _data=new T[_iSize];
    }
    
	void resizeLike(const Matrix<T>& other)
	{
		resize(other.rows(), other.cols());
	}
	
    T* data()
    {
        return _data;
    }

    const T* data() const
    {
        return _data;
    }

    void setConstant(T b)
    {
        std::fill(_data,_data+_iSize,b);
    }

	void setConstant(Eigen::Index iRows, Eigen::Index iColumns, T b)
	{
		resize(iRows, iColumns);
		setConstant(b);
	}

	void setZero()
	{
		setConstant(0.);
	}
    void setZero(Eigen::Index iRows, Eigen::Index iColumns)
    {
        resize(iRows,iColumns);
        setZero();
    }

	void setOnes()
	{
		setConstant(1.);
	}
	void setOnes(Eigen::Index iRows, Eigen::Index iColumns)
	{
		resize(iRows, iColumns);
		setOnes();
	}

    void setRandom()
	{
		setRandomUniform(*this, -1.f, 1.f);
	}

	void setRandom(Eigen::Index iRows, Eigen::Index iColumns)
	{
		resize(iRows, iColumns);
		setRandom();
	}

	Matrix<T> asDiagonal() const
	{
		Matrix<T> out;
		out.setZero(_iSize, _iSize);
		
		for (Eigen::Index i = 0; i < _iSize; i++)
			out(i, i) = _data[i];

		return out;
	}

    T& operator()(Eigen::Index iR, Eigen::Index iC)
    {
        assert(iR<_iRows);
        assert(iC<_iColumns);
        return *(_data+iR*_iColumns+iC);
    }
    
    const T& operator()(Eigen::Index iR, Eigen::Index iC) const
    {
        assert(iR<_iRows);
        assert(iC<_iColumns);
        return *(_data+iR*_iColumns+iC);
    }
    
    T& operator()(Eigen::Index iX)
    {
        assert(iX<_iSize);
        return *(_data+iX);
    }
    
    const T& operator()(Eigen::Index iX) const
    {
        assert(iX<_iSize);
        return *(_data+iX);
    }
    
    Matrix<T>& operator+=(const Matrix<T>& a)
    {
        assert(_iRows==a.rows());
        assert(_iColumns==a.cols());

        for(Eigen::Index i=0;i<_iSize;i++)
            _data[i]+=a(i);
        return *this;
    }
    
    Matrix<T> operator+( const Matrix<T>& a ) const
    {
        assert(_iRows==a.rows());
        assert(_iColumns==a.cols());

        return Matrix<T>(*this).operator+=(a);
    }

    Matrix<T>& operator+=(T d)
    {
        for(Eigen::Index i=0;i<_iSize;i++)
            _data[i]+=d;
        return *this;
    }
    Matrix<T> operator+(T d ) const
    {
        return Matrix<T>(*this).operator+=(d);
    }

    Matrix<T>& operator-=(const Matrix<T>& a)
    {
        assert(_iRows==a.rows());
        assert(_iColumns==a.cols());

        for(Eigen::Index i=0;i<_iSize;i++)
            _data[i]-=a(i);
        return *this;
    }

    Matrix<T> operator-( const Matrix<T>& a ) const
    {
        assert(_iRows==a.rows());
        assert(_iColumns==a.cols());

        return Matrix<T>(*this).operator-=(a);
    }

    Matrix<T>& operator-=(T d)
    {
        for(Eigen::Index i=0;i<_iSize;i++)
            _data[i]-=d;
        return *this;
    }

    Matrix<T> operator-(T d ) const
    {
        return Matrix<T>(*this).operator-=(d);
    }
    
    Matrix<T>& operator*=(T b)
    {
        for(Eigen::Index i=0;i<_iSize;i++)
            _data[i]*=b;

        return *this;
    }

    Matrix<T>& operator/=(T b)
    {
        for(Eigen::Index i=0;i<_iSize;i++)
            _data[i]/=b;

        return *this;
    }

    Matrix<T> operator/(T b) const // slow function!
    {
        return Matrix<T>(*this).operator/=(b);
    }

    Matrix<T> operator*(T b) const // slow function!
    {
        return Matrix<T>(*this).operator*=(b);
    }


////////////////////////////////////////////////////////
////////////////////////////////////////////////////////
////////////////////////////////////////////////////////
////////////////////////////////////////////////////////
//////////////////////////////////////////////
    Matrix<T>& operator*=(const Matrix<T>& b) // slow function!
    {
        assert(cols()==b.rows());

        Matrix<T> a(*this);
        resize(a._iRows,b._iColumns);


		// RKC algorithm
		Eigen::Index kMax=a._iColumns;
		setZero();

		for (Eigen::Index r = 0; r < _iRows; r++)
			for (Eigen::Index k = 0; k < kMax; k++)
			{ 
				auto af=a(r, k);
				for (Eigen::Index c = 0; c < _iColumns; c++)
					operator()(r,c)+= af * b(k, c);
			}
/*
		//RCK algorithm, slower
        for(Eigen::Index r=0;r<_iRows;r++)
        {
            for(Eigen::Index c=0;c<_iColumns;c++)
            {
                T temp=0.;

                for(Eigen::Index k=0;k<a._iColumns;k++)
                    temp+=a(r,k)*b(k,c);

                operator()(r,c)=temp;
            }
        }
	*/	
        return *this;
    }

    Matrix<T> cwiseProduct(const Matrix<T>& m) const
    {
        assert(cols()==m.cols());
        assert(rows()==m.rows());

        Matrix<T> out(*this);

        for(Eigen::Index i=0;i<_iSize;i++)
            out(i)*=m(i);

        return out;
    }

    Matrix<T> cwiseQuotient(const Matrix<T>& m) const
    {
        assert(cols()==m.cols());
        assert(rows()==m.rows());

        Matrix<T> out(*this);

        for(Eigen::Index i=0;i<_iSize;i++)
            out(i)/=m(i);

        return out;
    }

    Matrix<T> cwiseAbs() const
    {
        Matrix<T> out(*this);

        for(Eigen::Index i=0;i<_iSize;i++)
            out(i)=::abs(_data[i]);

        return out;
    }

	Matrix<T> cwiseSign() const
	{
		Matrix<T> out(*this);

		for (Eigen::Index i = 0; i < _iSize; i++)
			out(i) = std::copysign(1.f,_data[i]);

		return out;
	}

	Matrix<T> cwiseAbs2() const
    {
        Matrix<T> out(*this);

        for(Eigen::Index i=0;i<_iSize;i++)
            out(i)=_data[i]*_data[i]; //todo optimize

        return out;
    }

	Matrix<T> cube() const
	{
		Matrix<T> out(*this);

		for (Eigen::Index i = 0; i < _iSize; i++)
			out(i) = _data[i] * _data[i] * _data[i]; //todo optimize

		return out;
	}

	Matrix<T> log() const // applies on array only
	{
		Matrix<T> out(*this);

		for (Eigen::Index i = 0; i < _iSize; i++)
			out(i) = ::log(_data[i]);

		return out;
	}

	Matrix<T> round() const // applies on array only
	{
		Matrix<T> out(*this);

		for (Eigen::Index i = 0; i < _iSize; i++)
			out(i) = ::round(_data[i]);

		return out;
	}

	Matrix<T> cosh() const // applies on array only
	{
		Matrix<T> out(*this);

		for (Eigen::Index i = 0; i < _iSize; i++)
			out(i) = ::cosh(_data[i]);

		return out;
	}

	Matrix<T> tanh() const // applies on array only
	{
		Matrix<T> out(*this);

		for (Eigen::Index i = 0; i < _iSize; i++)
			out(i) = ::tanh(_data[i]);

		return out;
	}

	Matrix<T> exp() const // applies on array only
	{
		Matrix<T> out(*this);

		for (Eigen::Index i = 0; i < _iSize; i++)
			out(i) = ::exp(_data[i]);

		return out;
	}

	Matrix<T> cwiseSqrt() const
    {
        Matrix<T> out(*this);

        for(Eigen::Index i=0;i<_iSize;i++)
            out(i)=::sqrt(_data[i]);

        return out;
    }

	Matrix<T> square() const
	{
		Matrix<T> out(*this);

		for (Eigen::Index i = 0; i < _iSize; i++)
			out(i) =_data[i]* _data[i];

		return out;
	}

	Matrix<T> cwiseMin(T f) const
	{
		Matrix<T> out(*this);

		for (Eigen::Index i = 0; i < _iSize; i++)
			out(i) = std::min<T>(_data[i], f);

		return out;
	}
	
	Matrix<T> cwiseMin(const Matrix<T>& m) const
	{
		assert(m.rows() == rows());
		assert(m.cols() == cols());

		Matrix<T> out(*this);

		for (Eigen::Index i = 0; i < _iSize; i++)
			out(i) = std::min<T>(_data[i], m._data[i]);

		return out;
	}

	Matrix<T> cwiseMax(T f) const
    {
        Matrix<T> out(*this);

        for(Eigen::Index i=0;i<_iSize;i++)
            out(i)=std::max<T>(_data[i],f);

        return out;
    }

    Matrix<T> cwiseMax(const Matrix<T>& m) const
    {
        assert(m.rows()==rows());
        assert(m.cols()==cols());

        Matrix<T> out(*this);

        for(Eigen::Index i=0;i<_iSize;i++)
            out(i)=std::max<T>(_data[i],m._data[i]);

        return out;
    }

    T sum() const
    {
        T dSum=0.;
        for(Eigen::Index i=0;i<_iSize;i++)
            dSum+=_data[i];

        return dSum;
    }

	T squaredNorm() const
	{
		T dSumSq = 0.;
		for (Eigen::Index i = 0; i < _iSize; i++)
			dSumSq += _data[i]* _data[i];

		return dSumSq;
	}

	T norm() const
	{
		return ::sqrtf(squaredNorm());
	}

	T mean() const
	{
		return sum()/(T)_iSize;
	}
	
	T maxCoeff() const
    {
        if(_iSize==0)
            return 0.; //not clean

        T dMax=_data[0];
        for(Eigen::Index i=1;i<_iSize;i++)
            if(_data[i]>dMax)
                dMax=_data[i];

        return dMax;
    }

    Matrix<T> transpose() const // slow function!
    {
        Matrix<T> out(_iColumns,_iRows);

        for(Eigen::Index r=0;r<_iRows;r++)
            for(Eigen::Index c=0;c<_iColumns;c++)
                out(c,r)=operator()(r,c);

        return out;
    }

    Matrix<T> operator*(const Matrix<T>& a) const  // slow function!
    {
        return Matrix<T>(*this).operator*=(a);
    }

    Matrix<T> row(Eigen::Index iRow)
    {
        assert(iRow<_iRows);

        return Matrix<T>(_data+iRow*_iColumns,1,_iColumns);
    }

    const Matrix<T> row(Eigen::Index iRow) const
    {
        assert(iRow<_iRows);

        return Matrix<T>(_data+iRow*_iColumns,1,_iColumns);
    }

    Matrix<T> topRows(Eigen::Index iNbRow)
    {
        assert(iNbRow>=0);
        assert(iNbRow<_iRows);

        return Matrix<T>(_data,iNbRow,_iColumns);
    }

    const Matrix<T> topRows(Eigen::Index iNbRow) const
    {
        assert(iNbRow>=0);
        assert(iNbRow<_iRows);

        return Matrix<T>(_data,iNbRow,_iColumns);
    }

    Matrix<T> diagonal() const //slow!
    {
        Matrix<T> r(_iRows,1);

        for(Eigen::Index i=0;i<_iRows;i++)
            r(i)=operator()(i,i);

        return r;
    }

    T trace() const
    {
        T trace=(T)0;

        for(Eigen::Index i=0;i<_iRows;i++) //todo test square
            trace+=operator()(i,i);

        return trace;
    }

private:
    Eigen::Index _iRows,_iColumns,_iSize;
    T* _data;
    bool _bIsView;
};

typedef Matrix<float> MatrixFloat;
typedef Matrix<float> MatrixFloatView;

#endif


MatrixFloatView fromRawBuffer(MatrixFloat::Scalar* pBuffer, Eigen::Index iRows, Eigen::Index iCols);
const MatrixFloatView fromRawBufferConst(MatrixFloat::Scalar* pBuffer, Eigen::Index iRows, Eigen::Index iCols);
const MatrixFloatView fromRawBufferConst(const MatrixFloat::Scalar* pBuffer, Eigen::Index iRows, Eigen::Index iCols);
void copyInto(const MatrixFloat& mToCopy, MatrixFloat& m, Eigen::Index iStartRow);
float* rowPtr(MatrixFloat& m, Eigen::Index iRow);
const float* rowPtr(const MatrixFloat& m, Eigen::Index iRow);

MatrixFloatView createView(MatrixFloat& mRef);
const MatrixFloatView viewResize(const MatrixFloat& m, Eigen::Index iRows, Eigen::Index iCols);
const MatrixFloatView viewRow(const MatrixFloat& m, Eigen::Index iStartRow, Eigen::Index iEndRow); //create a row view starting at iStartRow to (not included) iEndRow
const MatrixFloat colExtract(const MatrixFloat& m, Eigen::Index iStartCol , Eigen::Index iEndCol);

MatrixFloat rowWiseSum(const MatrixFloat& m);
MatrixFloat rowWiseSumSq(const MatrixFloat& m);
MatrixFloat rowWiseAdd(const MatrixFloat& m, const MatrixFloat& d);
MatrixFloat rowWiseMult(const MatrixFloat& m, const MatrixFloat& d);
MatrixFloat rowWiseDivide(const MatrixFloat& m, const MatrixFloat& d);

MatrixFloat colWiseSum(const MatrixFloat& m);
MatrixFloat colWiseSumSq(const MatrixFloat& m);
MatrixFloat colWiseMean(const MatrixFloat& m);
MatrixFloat colWiseMin(const MatrixFloat& m);
MatrixFloat colWiseMax(const MatrixFloat& m);

void setRandomUniform(MatrixFloat& m, float fMin = -1.f, float fMax = 1.f);
void setRandomNormal(MatrixFloat& m, float fMean, float fNormal);
void setQuickBernoulli(MatrixFloat& m, float fProba); //quick bernoulli is 6x faster than ref bernoulli, resolution proba is 1/65536 
///////////////////////////////////////////////////////////////////////////
inline std::default_random_engine& randomEngine()
{
	static std::default_random_engine rng;
	return rng;
}
std::vector<Eigen::Index> randPerm(Eigen::Index iSize); //create a std::vector of Eigen::Index shuffled
void applyRowPermutation(const std::vector<Eigen::Index>& vPermutation, const MatrixFloat & mIn, MatrixFloat & mPermuted);
MatrixFloat decimate(const MatrixFloat& m, Eigen::Index iRatio);
Eigen::Index argmax(const MatrixFloat& m);
void rowsArgmax(const MatrixFloat& m, MatrixFloat& argM); //compute the argmax row by row
void clamp(MatrixFloat& m,float fClampMin,float fClampMax);
MatrixFloat tanh(const MatrixFloat& m);
MatrixFloat oneMinusSquare(const MatrixFloat& m);
void reverseData(float* pData, Eigen::Index iSize);

//4D tensor functions, access order in memory is: sample, channel, row, column
void channelWiseAdd(MatrixFloat& mIn,Eigen::Index iNbSamples,Eigen::Index iNbChannels,Eigen::Index iNbRows,Eigen::Index iNbCols,const MatrixFloat & weight);
MatrixFloat channelWiseMean(const MatrixFloat& m, Eigen::Index iNbSamples, Eigen::Index iNbChannels, Eigen::Index iNbRows, Eigen::Index iNbCols);

std::string toString(const MatrixFloat& m);
const MatrixFloat fromFile(const std::string& sFile);
const MatrixFloat fromString(const std::string& s);
bool toFile(const std::string& sFile, const MatrixFloat & m);

}
#endif
