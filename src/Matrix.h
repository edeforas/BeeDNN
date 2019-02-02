#ifndef Matrix_
#define Matrix_

#include <cassert>

//todo add more tests and optimize
// todo add optional typedef to eigen and update API

#ifdef USE_EIGEN

#include "Eigen/Core"
using namespace Eigen;
typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixFloat;

/*
const MatrixFloat without_last_row(MatrixFloat m)
{
    return m.block(0,0,m.rows()-1,m.cols()); //todo when rows==0
}

//template <class T>
const MatrixFloat without_last_column(MatrixFloat m)
{
    return m.block(0,0,m.rows(),m.cols()-1); //todo when rows==0
}
*/
/*
MatrixFloat concatHorizontally(const MatrixFloat& a, const MatrixFloat& b)
{
    MatrixFloat c(a.rows(), a.cols()+b.cols());
    c << a, b;
    return c;
}
*/
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

    Matrix<T>(size_t iRows,size_t iColumns)
    {
        _iRows=iRows;
        _iColumns=iColumns;
        _iSize=_iRows*_iColumns;
        _data=new T[_iSize];
        _bDelete=true;
    }
    
    Matrix<T>(T* pData,size_t iRows,size_t iColumns)
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

        for(unsigned int i=0;i<size();i++)
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
        
        for(unsigned int i=0;i<size();i++)
            operator()(i)=b(i);
        
        return *this;
    }
    
    size_t rows() const
    {
        return _iRows;
    }

    size_t cols() const
    {
        return _iColumns;
    }
    
    size_t size() const
    {
        return _iSize;
    }

    void resize(size_t iRows,size_t iColumns) // slow function!
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
        for(unsigned int i=0;i<_iSize;i++)
            _data[i]=b;
    }

    void setZero()
    {
        setConstant(0.);
    }

    T& operator()(size_t iR,size_t iC)
    {
        assert(iR<_iRows);
        assert(iC<_iColumns);
        return *(_data+iR*_iColumns+iC);
    }
    
    const T& operator()(size_t iR,size_t iC) const
    {
        assert(iR<_iRows);
        assert(iC<_iColumns);
        return *(_data+iR*_iColumns+iC);
    }
    
    T& operator()(size_t iX)
    {
        assert(iX<_iSize);
        return *(_data+iX);
    }
    
    const T& operator()(size_t iX) const
    {
        assert(iX<_iSize);
        return *(_data+iX);
    }
    
    Matrix<T>& operator+=(const Matrix<T>& a)
    {
        assert(_iRows==a.rows());
        assert(_iColumns==a.cols());

        for(size_t i=0;i<_iSize;i++)
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
        for(size_t i=0;i<_iSize;i++)
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

        for(unsigned int i=0;i<_iSize;i++)
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
        for(unsigned int i=0;i<_iSize;i++)
            _data[i]-=d;
        return *this;
    }
    Matrix<T> operator-(T d ) const
    {
        return Matrix<T>(*this).operator-=(d);
    }
    
    Matrix<T>& operator*=(T b)
    {
        for(unsigned int i=0;i<_iSize;i++)
            _data[i]*=b;

        return *this;
    }

    Matrix<T>& operator/=(T b)
    {
        for(unsigned int i=0;i<_iSize;i++)
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

        for(unsigned int r=0;r<_iRows;r++)
        {
            for(unsigned int c=0;c<_iColumns;c++)
            {
                T temp=0.;

                for(unsigned int k=0;k<a._iColumns;k++)
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

        for(unsigned int i=0;i<_iSize;i++)
            out(i)*=m(i);

        return out;
    }

    Matrix<T> cwiseDivide(const Matrix<T>& m) const
    {
        assert(cols()==m.cols());
        assert(rows()==m.rows());

        Matrix<T> out(*this);

        for(unsigned int i=0;i<_iSize;i++)
            out(i)/=m(i);

        return out;
    }

    Matrix<T> scalar_mult(T d) const
    {
        Matrix<T> out(*this);

        for(unsigned int i=0;i<_iSize;i++)
            out(i)*=d;

        return out;
    }

    T sum() const
    {
        T dSum=0.;
        for(unsigned int i=0;i<_iSize;i++)
            dSum+=_data[i];

        return dSum;
    }

    T max() const
    {
        if(_iSize==0)
            return 0.; //not clean

        T dMax=_data[0];
        for(unsigned int i=1;i<_iSize;i++)
            if(_data[i]>dMax)
                dMax=_data[i];

        return dMax;
    }
	
    Matrix<T> transpose() const // slow function!
    {
        Matrix<T> out(_iColumns,_iRows);

        for(unsigned int r=0;r<_iRows;r++)
            for(unsigned int c=0;c<_iColumns;c++)
                out(c,r)=operator()(r,c);

        return out;
    }

    Matrix<T> operator*(const Matrix<T>& a) const  // slow function!
    {
        return Matrix<T>(*this).operator*=(a);
    }


    Matrix<T> row(size_t iRow)
    {
        assert(iRow<_iRows);

        return Matrix<T>(_data+iRow*_iColumns,1,_iColumns);
    }

    
    const Matrix<T> row(size_t iRow) const
    {
        assert(iRow<_iRows);

        return Matrix<T>(_data+iRow*_iColumns,1,_iColumns);
    }


    Matrix<T> row_sum() const
    {
        Matrix<T> r(_iRows,1);

        for(unsigned int i=0;i<_iRows;i++)
            r(i)=row(i).sum();

        return r;
    }

    Matrix<T> diag() const
    {
        Matrix<T> r(_iRows,1);

        for(unsigned int i=0;i<_iRows;i++)
            r(i)=operator()(i,i);

        return r;
    }
    bool is_vector() const
    {
        return (_iRows==1) || (_iColumns==1);
    }
    
private:
    size_t _iRows,_iColumns,_iSize;
    T* _data;
    bool _bDelete;
};

typedef Matrix<float> MatrixFloat;




/*
template <class T>
const Matrix<T> without_last_row(const Matrix<T>& m)
{
    return from_raw_buffer((T*)m.data(),m.rows()-1,m.cols());
}


template <class T>
const Matrix<T> without_last_column(const Matrix<T>& a) // slow function!
{
    assert(a.cols()>0);

    Matrix<T> m(a.rows(), a.cols()-1);

    for(unsigned int r=0;r<a.rows();r++)
        for(unsigned int c=0;c<a.cols()-1;c++)
            m(r,c)=a(r,c);

    return m;
}

template <class T>
Matrix<T> concatHorizontally(const Matrix<T> & a, const Matrix<T> & b) // slow function!
{
    //concat horizontally
    size_t r=a.rows();
    size_t c=a.cols();

    assert(r==b.rows());

    Matrix<T> out(r,c+b.cols());

    for(unsigned int j=0;j<r;j++)
        for(unsigned int i=0;i<c;i++)
            out(j,i)=a(j,i);

    for(unsigned int j=0;j<r;j++)
        for(unsigned int i=0;i<b.cols();i++)
            out(j,i+c)=b(j,i);

    return out;
}
*/
#endif

const MatrixFloat from_raw_buffer(float *pBuffer,size_t iRows,size_t iCols);

#endif
