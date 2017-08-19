#ifndef _Matrix_
#define _Matrix_

#include <cassert>

//todo add more tests and optimize

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

    Matrix<T>(unsigned int iRows,unsigned int iColumns=1)
    {
        _iRows=iRows;
        _iColumns=iColumns;
        _iSize=_iRows*_iColumns;
        _data=new T[_iSize];
        _bDelete=true;
    }
    
    Matrix<T>(T* pData,unsigned int iRows,unsigned int iColumns)
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
    
    Matrix<T>& operator=( const Matrix<T>& b)
    {
        resize(b.rows(),b.columns());
        
        for(unsigned int i=0;i<size();i++)
            operator()(i)=b(i);
        
        return *this;
    }
    
    unsigned int rows() const
    {
        return _iRows;
    }

    unsigned int columns() const
    {
        return _iColumns;
    }
    
    unsigned int size() const
    {
        return _iSize;
    }

    void resize(unsigned int iRows,unsigned int iColumns) // slow function!
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

    void set_constant(T b)
    {
        for(unsigned int i=0;i<_iSize;i++)
            _data[i]=b;
    }

    void set_zero()
    {
        set_constant(0.);
    }

    T& operator()(unsigned int iR,unsigned int iC)
    {
        assert(iR<_iRows);
        assert(iC<_iColumns);
        return *(_data+iR*_iColumns+iC);
    }
    
    const T& operator()(unsigned int iR,unsigned int iC) const
    {
        assert(iR<_iRows);
        assert(iC<_iColumns);
        return *(_data+iR*_iColumns+iC);
    }
    
    T& operator()(unsigned int iX)
    {
        assert(iX<_iSize);
        return *(_data+iX);
    }
    
    const T& operator()(unsigned int iX) const
    {
        assert(iX<_iSize);
        return *(_data+iX);
    }
    
    Matrix<T>& operator+=(const Matrix<T>& a)
    {
        assert(_iRows==a.rows());
        assert(_iColumns==a.columns());

        for(unsigned int i=0;i<_iSize;i++)
            _data[i]+=a(i);
        return *this;
    }
    
    Matrix<T> operator+( const Matrix<T>& a ) const
    {
        assert(_iRows==a.rows());
        assert(_iColumns==a.columns());

        return Matrix<T>(*this).operator+=(a);
    }

    Matrix<T>& operator+=(T d)
    {
        for(unsigned int i=0;i<_iSize;i++)
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
        assert(_iColumns==a.columns());

        for(unsigned int i=0;i<_iSize;i++)
            _data[i]-=a(i);
        return *this;
    }
    Matrix<T> operator-( const Matrix<T>& a ) const
    {
        assert(_iRows==a.rows());
        assert(_iColumns==a.columns());

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
        assert(columns()==b.rows());

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

    Matrix<T> element_product(const Matrix<T>& m) const
    {
        assert(columns()==m.columns());
        assert(rows()==m.rows());

        Matrix<T> out(*this);

        for(unsigned int i=0;i<_iSize;i++)
            out(i)*=m(i);

        return out;
    }

    Matrix<T> element_divide(const Matrix<T>& m) const
    {
        assert(columns()==m.columns());
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

    Matrix<T> concat(const Matrix<T> & b) // slow function!
    {
        assert(b.rows()==rows());

        Matrix<T> mT(_iRows,_iColumns+b._iColumns);

        for(unsigned int r=0;r<_iRows;r++)
            for(unsigned int c=0;c<_iColumns;c++)
                mT(r,c)=operator()(r,c);

        for(unsigned int r=0;r<b.rows();r++)
            for(unsigned int c=0;c<b.columns();c++)
                mT(r,c+_iColumns)=b(r,c);

        return mT;
    }

    Matrix<T> operator*(const Matrix<T>& a) const  // slow function!
    {
        return Matrix<T>(*this).operator*=(a);
    }


    Matrix<T> row(unsigned int iRow)
    {
        assert(iRow<_iRows);

        return Matrix<T>(_data+iRow*_iColumns,1,_iColumns);
    }

    
    const Matrix<T> row(unsigned int iRow) const
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

    const Matrix<T> without_last_row() const
    {
        assert(_iRows>0);
        return Matrix<T>(_data,_iRows-1,_iColumns);
    }
    
    const Matrix<T> without_last_column() const // slow function!
    {
        assert(_iColumns>0);

        Matrix<T> m(_iRows,_iColumns-1);

        for(unsigned int r=0;r<_iRows;r++)
            for(unsigned int c=0;c<_iColumns-1;c++)
                m(r,c)=operator()(r,c);

        return m;
    }

    bool is_vector() const
    {
        return (_iRows==1) || (_iColumns==1);
    }
    
private:
    unsigned int _iRows,_iColumns,_iSize;
    T* _data;
    bool _bDelete;
};


typedef Matrix<double> MatrixDouble;
typedef Matrix<float> MatrixFloat;

#endif
