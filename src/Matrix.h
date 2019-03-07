#ifndef Matrix_
#define Matrix_

#include <cassert>
#include <string>
using namespace std;

//todo add more tests and optimize

#ifdef USE_EIGEN

#include "Eigen/Core"
using namespace Eigen;
typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixFloat;

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
        for(int i=0;i<_iSize;i++)
            _data[i]=b;
    }

    void setZero()
    {
        setConstant(0.);
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

    Matrix<T> cwiseAbs2() const
    {
        Matrix<T> out(*this);

        for(int i=0;i<_iSize;i++)
            out(i)=_data[i]*_data[i]; //todo optimize

        return out;
    }

    T sum() const
    {
        T dSum=0.;
        for(int i=0;i<_iSize;i++)
            dSum+=_data[i];

        return dSum;
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

    Matrix<T> diagonal() const //slow!
    {
        Matrix<T> r(_iRows,1);

        for(int i=0;i<_iRows;i++)
            r(i)=operator()(i,i);

        return r;
    }

private:
    int _iRows,_iColumns,_iSize;
    T* _data;
    bool _bDelete;
};

typedef Matrix<float> MatrixFloat;

#endif

const MatrixFloat from_raw_buffer(const float *pBuffer,int iRows,int iCols);
MatrixFloat rowWiseSum(const MatrixFloat& m);
MatrixFloat rand_perm(int iSize); //create a vector of index shuffled
MatrixFloat decimate(const MatrixFloat& m, int iRatio);
int argmax(const MatrixFloat& m);
string matrix_to_string(const MatrixFloat& m);
void contatenateVerticallyInto(const MatrixFloat& mA, const MatrixFloat& mB, MatrixFloat& mAB);
const MatrixFloat withoutLastRow(const MatrixFloat& m);
MatrixFloat lastRow( MatrixFloat& m);
const MatrixFloat lastRow(const MatrixFloat& m);
const MatrixFloat addColumnOfOne(const MatrixFloat& m);

#endif

