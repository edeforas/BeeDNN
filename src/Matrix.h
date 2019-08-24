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
#include <cmath>
using namespace std;

#ifdef USE_EIGEN

#include "Eigen/Core"
using namespace Eigen;
typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixFloat;
typedef Eigen::Map<MatrixFloat> MatrixFloatView;

#else

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
        _bDelete=false;
    }

    Matrix<T>(int iRows,int iColumns)
    {
        _iRows=iRows;
        _iColumns=iColumns;
        _iSize=_iRows*_iColumns;
        _data=new T[_iSize];
        _bDelete=true;
    }
    
    Matrix<T>(T* pData,int iRows,int iColumns)
    {
        _iRows=iRows;
        _iColumns=iColumns;
        _iSize=_iRows*_iColumns;
        _data=pData;
        _bDelete=false;
    }
    
    static const Matrix<T> from_raw_buffer(const T* pData,int iRows,int iColumns)
    {
        Matrix<T> m;

        m._iRows=iRows;
        m._iColumns=iColumns;
        m._iSize=iRows*iColumns;
        m._data=(T*)pData;
        m._bDelete=false;

        return m;
    }

    Matrix<T>(const Matrix<T> &a)
    {
        _iRows=a._iRows;
        _iColumns=a._iColumns;
        _iSize=_iRows*_iColumns;
        _data=new T[_iSize];
        _bDelete=true;

        for( int i=0;i<size();i++)
            _data[i]=a(i);
        //todo use or merge with operator=()(a); ??
    }

    ~Matrix<T>()
    {
        if(_bDelete)
            delete [] _data;
    }

    void assign(T* first,T* last)
    {
        resize(1,(unsigned int)(last-first));
        for(unsigned int i=0;i<size();i++)
            operator()(i)=*first++;

        //todo  check and optimize
    }
    
    Matrix<T>& operator=( const Matrix<T>& b)
    {
        resize(b.rows(),b.cols());
        
        for(int i=0;i<size();i++)
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

	Matrix<T>& operator-()
	{
		for (int i = 0; i < size(); i++)
			_data[i] = -_data[i];

		return *this;
	}
	
	int rows() const
    {
        return _iRows;
    }

    int cols() const
    {
        return _iColumns;
    }
    
    int size() const
    {
        return _iSize;
    }

    void resize(int iRows,int iColumns) // slow function!
    {
        if((iColumns==_iColumns) && ( iRows==_iRows))
            return;

        _iRows=iRows;
        _iColumns=iColumns;
        _iSize=_iRows*_iColumns;
        
        if(_bDelete)
        {
            delete[] _data;
        }
        else
            _bDelete=true;

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

	void setZero()
	{
		setConstant(0.);
	}

	void setOnes()
	{
		setConstant(1.);
	}
    void setZero(int iRows,int iColumns)
    {
        resize(iRows,iColumns);
        setZero();
    }

    void setRandom()
	{
	    for(int i=0;i<_iSize;i++)
            _data[i]=((T)rand()/(T)RAND_MAX-0.5f)*2.f;
	}

	void setRandom(int iRows, int iColumns)
	{
		resize(iRows, iColumns);
		setRandom();
	}

	Matrix<T> asDiagonal() const
	{
		Matrix<T> out;
		out.setZero(_iSize, _iSize);
		
		for (int i = 0; i < _iSize; i++)
			out(i, i) = _data[i];

		return out;
	}

    T& operator()(int iR,int iC)
    {
        assert(iR<_iRows);
        assert(iC<_iColumns);
        return *(_data+iR*_iColumns+iC);
    }
    
    const T& operator()(int iR,int iC) const
    {
        assert(iR<_iRows);
        assert(iC<_iColumns);
        return *(_data+iR*_iColumns+iC);
    }
    
    T& operator()(int iX)
    {
        assert(iX<_iSize);
        return *(_data+iX);
    }
    
    const T& operator()(int iX) const
    {
        assert(iX<_iSize);
        return *(_data+iX);
    }
    
    Matrix<T>& operator+=(const Matrix<T>& a)
    {
        assert(_iRows==a.rows());
        assert(_iColumns==a.cols());

        for(int i=0;i<_iSize;i++)
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
        for(int i=0;i<_iSize;i++)
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

        for(int i=0;i<_iSize;i++)
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
        for(int i=0;i<_iSize;i++)
            _data[i]-=d;
        return *this;
    }

    Matrix<T> operator-(T d ) const
    {
        return Matrix<T>(*this).operator-=(d);
    }
    
    Matrix<T>& operator*=(T b)
    {
        for(int i=0;i<_iSize;i++)
            _data[i]*=b;

        return *this;
    }

    Matrix<T>& operator/=(T b)
    {
        for(int i=0;i<_iSize;i++)
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

    Matrix<T>& operator*=(const Matrix<T>& b) // slow function!
    {
        assert(cols()==b.rows());

        Matrix<T> a(*this);
        resize(a._iRows,b._iColumns);

        for(int r=0;r<_iRows;r++)
        {
            for(int c=0;c<_iColumns;c++)
            {
                T temp=0.;

                for(int k=0;k<a._iColumns;k++)
                    temp+=a(r,k)*b(k,c);

                operator()(r,c)=temp;
            }
        }

        return *this;
    }

    Matrix<T> cwiseProduct(const Matrix<T>& m) const
    {
        assert(cols()==m.cols());
        assert(rows()==m.rows());

        Matrix<T> out(*this);

        for(int i=0;i<_iSize;i++)
            out(i)*=m(i);

        return out;
    }

    Matrix<T> cwiseQuotient(const Matrix<T>& m) const
    {
        assert(cols()==m.cols());
        assert(rows()==m.rows());

        Matrix<T> out(*this);

        for(int i=0;i<_iSize;i++)
            out(i)/=m(i);

        return out;
    }

    Matrix<T> cwiseAbs() const
    {
        Matrix<T> out(*this);

        for(int i=0;i<_iSize;i++)
            out(i)=fabs(_data[i]);

        return out;
    }

	Matrix<T> cwiseAbs2() const
    {
        Matrix<T> out(*this);

        for(int i=0;i<_iSize;i++)
            out(i)=_data[i]*_data[i]; //todo optimize

        return out;
    }

	Matrix<T> cwiseSign() const
	{
		Matrix<T> out(*this);

		for (int i = 0; i < _iSize; i++)
			out(i) = (_data[i] >= 0.f)*2.f -1.f;

		return out;
	}

	Matrix<T> log() const //todo check applies on an array only
	{
		Matrix<T> out(*this);

		for (int i = 0; i < _iSize; i++)
			out(i) = logf(_data[i]);

		return out;
	}

	Matrix<T> cosh() const //todo check applies on an array only
	{
		Matrix<T> out(*this);

		for (int i = 0; i < _iSize; i++)
			out(i) = coshf(_data[i]);

		return out;
	}

	Matrix<T> tanh() const //todo check applies on an array only
	{
		Matrix<T> out(*this);

		for (int i = 0; i < _iSize; i++)
			out(i) = tanhf(_data[i]);

		return out;
	}

	Matrix<T> exp() const //todo check applies on an array only
	{
		Matrix<T> out(*this);

		for (int i = 0; i < _iSize; i++)
			out(i) = expf(_data[i]);

		return out;
	}

	Matrix<T> cwiseSqrt() const
    {
        Matrix<T> out(*this);

        for(int i=0;i<_iSize;i++)
            out(i)=sqrtf(_data[i]);

        return out;
    }

	Matrix<T> cwiseMin(T f) const
	{
		Matrix<T> out(*this);

		for (int i = 0; i < _iSize; i++)
			out(i) = std::min<T>(_data[i], f);

		return out;
	}
	
	Matrix<T> cwiseMin(const Matrix<T>& m) const
	{
		assert(m.rows() == rows());
		assert(m.cols() == cols());

		Matrix<T> out(*this);

		for (int i = 0; i < _iSize; i++)
			out(i) = std::min<T>(_data[i], m._data[i]);

		return out;
	}

	Matrix<T> cwiseMax(T f) const
    {
        Matrix<T> out(*this);

        for(int i=0;i<_iSize;i++)
            out(i)=std::max<T>(_data[i],f);

        return out;
    }

    Matrix<T> cwiseMax(const Matrix<T>& m) const
    {
        assert(m.rows()==rows());
        assert(m.cols()==cols());

        Matrix<T> out(*this);

        for(int i=0;i<_iSize;i++)
            out(i)=std::max<T>(_data[i],m._data[i]);

        return out;
    }

    T sum() const
    {
        T dSum=0.;
        for(int i=0;i<_iSize;i++)
            dSum+=_data[i];

        return dSum;
    }

	T mean() const
	{
		return sum()/(float)_iSize;
	}
	
	T maxCoeff() const
    {
        if(_iSize==0)
            return 0.; //not clean

        T dMax=_data[0];
        for(int i=1;i<_iSize;i++)
            if(_data[i]>dMax)
                dMax=_data[i];

        return dMax;
    }

    Matrix<T> transpose() const // slow function!
    {
        Matrix<T> out(_iColumns,_iRows);

        for(int r=0;r<_iRows;r++)
            for(int c=0;c<_iColumns;c++)
                out(c,r)=operator()(r,c);

        return out;
    }

    Matrix<T> operator*(const Matrix<T>& a) const  // slow function!
    {
        return Matrix<T>(*this).operator*=(a);
    }

    Matrix<T> row(int iRow)
    {
        assert(iRow<_iRows);

        return Matrix<T>(_data+iRow*_iColumns,1,_iColumns);
    }

    const Matrix<T> row(int iRow) const
    {
        assert(iRow<_iRows);

        return Matrix<T>(_data+iRow*_iColumns,1,_iColumns);
    }

    Matrix<T> topRows(int iNbRow)
    {
        assert(iNbRow>=0);
        assert(iNbRow<_iRows);

        return Matrix<T>(_data,iNbRow,_iColumns);
    }

    const Matrix<T> topRows(int iNbRow) const
    {
        assert(iNbRow>=0);
        assert(iNbRow<_iRows);

        return Matrix<T>(_data,iNbRow,_iColumns);
    }

    Matrix<T> diagonal() const //slow!
    {
        Matrix<T> r(_iRows,1);

        for(int i=0;i<_iRows;i++)
            r(i)=operator()(i,i);

        return r;
    }

    T trace() const
    {
        T trace=0;

        for(int i=0;i<_iRows;i++) //todo test square
            trace+=operator()(i,i);

        return trace;
    }

private:
    int _iRows,_iColumns,_iSize;
    T* _data;
    bool _bDelete;
};

typedef Matrix<float> MatrixFloat;
typedef Matrix<float> MatrixFloatView;

#endif

MatrixFloatView fromRawBuffer(float *pBuffer, int iRows, int iCols);
const MatrixFloatView fromRawBuffer(const float *pBuffer, int iRows, int iCols);
MatrixFloat rowWiseSum(const MatrixFloat& m);
MatrixFloat rowWiseAdd(const MatrixFloat& m, const MatrixFloat& d);
MatrixFloat rowWiseDivide(const MatrixFloat& m, const MatrixFloat& d);
void arraySub(MatrixFloat& m,float f);
MatrixFloat randPerm(int iSize); //create a vector of index shuffled
void applyRowPermutation(const MatrixFloat & mPermutationIndex, const MatrixFloat & mIn, MatrixFloat & mPermuted); 
const MatrixFloat rowRange(const MatrixFloat& m, int iStartRow, int iEndRow); //create a row view starting at iStartRow to (not included) iEndRow
MatrixFloat decimate(const MatrixFloat& m, int iRatio);
int argmax(const MatrixFloat& m);
void labelToOneHot(const MatrixFloat& mLabel, MatrixFloat& mOneMat, int iNbClass=0);
void rowsArgmax(const MatrixFloat& m, MatrixFloat& argM); //compute the argmax row by row
void contatenateVerticallyInto(const MatrixFloat& mA, const MatrixFloat& mB, MatrixFloat& mAB);
const MatrixFloat addColumnOfOne(const MatrixFloat& m);

string toString(const MatrixFloat& m);
const MatrixFloat fromFile(const string& sFile);
const MatrixFloat fromString(const string& s);
bool toFile(const string& sFile, const MatrixFloat & m);

#endif

