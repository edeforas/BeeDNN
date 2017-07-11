#ifndef _Matrix_
#define _Matrix_

#include <assert.h>

//todo add more tests and optimize

class Matrix
{
public:
    Matrix()
    {
        _iRows=0;
        _iColumns=0;
        _iSize=_iRows*_iColumns;
        _data=0;
        _bDelete=false;
    }

    Matrix(int iRows,int iColumns)
    {
        assert(iColumns>=0);
        assert(iRows>=0);
        
        _iRows=iRows;
        _iColumns=iColumns;
        _iSize=_iRows*_iColumns;
        _data=new double[_iSize];
        _bDelete=true;
    }
    
    Matrix(double* pData,int iRows,int iColumns)
    {
        assert(iColumns>=0);
        assert(iRows>=0);
        
        _iRows=iRows;
        _iColumns=iColumns;
        _iSize=_iRows*_iColumns;
        _data=pData;
        _bDelete=false;
    }
    
    Matrix(const Matrix &a)
    {
        _iRows=a._iRows;
        _iColumns=a._iColumns;
        _iSize=_iRows*_iColumns;
        _data=new double[_iSize];
        _bDelete=true;

        for(int i=0;i<size();i++)
            _data[i]=a(i);
    //todo use or merge with operator=()(a); ??
    }

    ~Matrix()
    {
        if(_bDelete)
            delete [] _data;
    }
    
    Matrix& operator=( const Matrix& b)
    {
        resize(b.rows(),b.columns());
        
        for(int i=0;i<size();i++)
            operator()(i)=b(i);
        
        return *this;
    }
    
    int rows() const
    {
        return _iRows;
    }

    int columns() const
    {
        return _iColumns;
    }
    
    void resize(int iRows,int iColumns)
    {
        assert(iColumns>=0);
        assert(iRows>=0);

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

        _data=new double[_iSize];
    }
    
    void setConstant(double b)
    {
        for(int i=0;i<_iSize;i++)
            _data[i]=b;
    }

    void setZero()
    {
        setConstant(0);
    }

    double& operator()(int iR,int iC)
    {
        assert(iR>=0);
        assert(iC>=0);
        assert(iR<_iRows);
        assert(iC<_iColumns);
        return *(_data+iR*_iColumns+iC);
    }
    
    const double& operator()(int iR,int iC) const
    {
        assert(iR>=0);
        assert(iC>=0);
        assert(iR<_iRows);
        assert(iC<_iColumns);
        return *(_data+iR*_iColumns+iC);
    }
    
    double& operator()(int iX)
    {
        assert(iX>=0);
        assert(iX<_iSize);
        return *(_data+iX);
    }
    
    const double& operator()(int iX) const
    {
        assert(iX>=0);
        assert(iX<_iSize);
        return *(_data+iX);
    }
    
    Matrix& operator+=(const Matrix& a)
    {
        assert(_iRows==a.rows());
        assert(_iColumns==a.columns());

        for(int i=0;i<_iSize;i++)
            _data[i]+=a(i);
        return *this;
    }
    
    Matrix operator+( const Matrix& a ) const
    {
        assert(_iRows==a.rows());
        assert(_iColumns==a.columns());

        return Matrix(*this).operator+=(a);
    }

    Matrix& operator-=(const Matrix& a)
    {
        assert(_iRows==a.rows());
        assert(_iColumns==a.columns());

        for(int i=0;i<_iSize;i++)
            _data[i]-=a(i);
        return *this;
    }

    Matrix operator-( const Matrix& a ) const
    {
        assert(_iRows==a.rows());
        assert(_iColumns==a.columns());

        return Matrix(*this).operator-=(a);
    }
    
    Matrix& operator*=(double b)
    {
        for(int i=0;i<_iSize;i++)
            _data[i]*=b;

        return *this;
    }

    Matrix operator*(double b) const
    {
        return Matrix(*this).operator*=(b);
    }

    Matrix& operator*=(const Matrix& b)
    {
        assert(columns()==b.rows());

        Matrix a(*this);
        resize(a._iRows,b._iColumns);

        for(int r=0;r<_iRows;r++)
        {
            for(int c=0;c<_iColumns;c++)
            {
                double temp=0.;

                for(int k=0;k<a._iColumns;k++)
                    temp+=a(r,k)*b(k,c);

                operator()(r,c)=temp;
            }
        }

        return *this;
    }

    Matrix elementProduct(const Matrix& m) const
    {
        assert(columns()==m.columns());
        assert(rows()==m.rows());

        Matrix out(*this);

        for(int i=0;i<_iSize;i++)
            out(i)*=m(i);

        return out;
    }

    Matrix scalarMult(double d) const
    {
        Matrix out(*this);

        for(int i=0;i<_iSize;i++)
            out(i)*=d;

        return out;
    }

    Matrix transpose() const
    {
        Matrix out(_iColumns,_iRows);

        for(int r=0;r<_iRows;r++)
            for(int c=0;c<_iColumns;c++)
                out(c,r)=operator()(r,c);

        return out;
    }

    Matrix concat(const Matrix & b)
    {
        assert(b.rows()==rows());

        Matrix mT(_iRows,_iColumns+b._iColumns);

        for(int r=0;r<_iRows;r++)
            for(int c=0;c<_iColumns;c++)
                mT(r,c)=operator()(r,c);

        for(int r=0;r<b.rows();r++)
            for(int c=0;c<b.columns();c++)
                mT(r,c+_iColumns)=b(r,c);

        return mT;
    }

    Matrix operator*(const Matrix& a) const
    {
        return Matrix(*this).operator*=(a);
    }
    
    const Matrix row(int iRow) const
    {
        assert(iRow>=0);
        assert(iRow<_iRows);

        return Matrix(_data+iRow*_iColumns,1,_iColumns);
    }
    
    const Matrix without_last_row() const
    {
        assert(_iRows>0);
        return Matrix(_data,_iRows-1,_iColumns);
    }
    
    const Matrix without_last_column() const
    {
        assert(_iColumns>0);

        Matrix m(_iRows,_iColumns-1);

        for(int r=0;r<_iRows;r++)
            for(int c=0;c<_iColumns-1;c++)
                m(r,c)=operator()(r,c);

        return m;
    }

    int size() const
    {
        return _iSize;
    }
    
private:
    int _iRows,_iColumns,_iSize;
    double* _data;
    bool _bDelete;
};

#endif
