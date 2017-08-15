#ifndef _Matrix_
#define _Matrix_

#include <cassert>

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

    Matrix(unsigned int iRows,unsigned int iColumns=1)
    {
        _iRows=iRows;
        _iColumns=iColumns;
        _iSize=_iRows*_iColumns;
        _data=new double[_iSize];
        _bDelete=true;
    }
    
    Matrix(double* pData,unsigned int iRows,unsigned int iColumns)
    {
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

        for(unsigned int i=0;i<size();i++)
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

        _data=new double[_iSize];
    }
    
    double* data()
    {
        return _data;
    }

    const double* data() const
    {
        return _data;
    }

    void set_constant(double b)
    {
        for(unsigned int i=0;i<_iSize;i++)
            _data[i]=b;
    }

    void set_zero()
    {
        set_constant(0.);
    }

    double& operator()(unsigned int iR,unsigned int iC)
    {
        assert(iR<_iRows);
        assert(iC<_iColumns);
        return *(_data+iR*_iColumns+iC);
    }
    
    const double& operator()(unsigned int iR,unsigned int iC) const
    {
        assert(iR<_iRows);
        assert(iC<_iColumns);
        return *(_data+iR*_iColumns+iC);
    }
    
    double& operator()(unsigned int iX)
    {
        assert(iX<_iSize);
        return *(_data+iX);
    }
    
    const double& operator()(unsigned int iX) const
    {
        assert(iX<_iSize);
        return *(_data+iX);
    }
    
    Matrix& operator+=(const Matrix& a)
    {
        assert(_iRows==a.rows());
        assert(_iColumns==a.columns());

        for(unsigned int i=0;i<_iSize;i++)
            _data[i]+=a(i);
        return *this;
    }
    
    Matrix operator+( const Matrix& a ) const
    {
        assert(_iRows==a.rows());
        assert(_iColumns==a.columns());

        return Matrix(*this).operator+=(a);
    }

    Matrix& operator+=(double d)
    {
        for(unsigned int i=0;i<_iSize;i++)
            _data[i]+=d;
        return *this;
    }
    Matrix operator+( double d ) const
    {
        return Matrix(*this).operator+=(d);
    }

    Matrix& operator-=(const Matrix& a)
    {
        assert(_iRows==a.rows());
        assert(_iColumns==a.columns());

        for(unsigned int i=0;i<_iSize;i++)
            _data[i]-=a(i);
        return *this;
    }
    Matrix operator-( const Matrix& a ) const
    {
        assert(_iRows==a.rows());
        assert(_iColumns==a.columns());

        return Matrix(*this).operator-=(a);
    }

    Matrix& operator-=(double d)
    {
        for(unsigned int i=0;i<_iSize;i++)
            _data[i]-=d;
        return *this;
    }
    Matrix operator-( double d ) const
    {
        return Matrix(*this).operator-=(d);
    }
    
    Matrix& operator*=(double b)
    {
        for(unsigned int i=0;i<_iSize;i++)
            _data[i]*=b;

        return *this;
    }

    Matrix& operator/=(double b)
    {
        for(unsigned int i=0;i<_iSize;i++)
            _data[i]/=b;

        return *this;
    }

    Matrix operator/(double b) const // slow function!
    {
        return Matrix(*this).operator/=(b);
    }

    Matrix operator*(double b) const // slow function!
    {
        return Matrix(*this).operator*=(b);
    }

    Matrix& operator*=(const Matrix& b) // slow function!
    {
        assert(columns()==b.rows());

        Matrix a(*this);
        resize(a._iRows,b._iColumns);

        for(unsigned int r=0;r<_iRows;r++)
        {
            for(unsigned int c=0;c<_iColumns;c++)
            {
                double temp=0.;

                for(unsigned int k=0;k<a._iColumns;k++)
                    temp+=a(r,k)*b(k,c);

                operator()(r,c)=temp;
            }
        }

        return *this;
    }

    Matrix element_product(const Matrix& m) const
    {
        assert(columns()==m.columns());
        assert(rows()==m.rows());

        Matrix out(*this);

        for(unsigned int i=0;i<_iSize;i++)
            out(i)*=m(i);

        return out;
    }

    Matrix element_divide(const Matrix& m) const
    {
        assert(columns()==m.columns());
        assert(rows()==m.rows());

        Matrix out(*this);

        for(unsigned int i=0;i<_iSize;i++)
            out(i)/=m(i);

        return out;
    }

    Matrix scalar_mult(double d) const
    {
        Matrix out(*this);

        for(unsigned int i=0;i<_iSize;i++)
            out(i)*=d;

        return out;
    }

	double sum() const
    {
		double dSum=0.;
        for(unsigned int i=0;i<_iSize;i++)
            dSum+=_data[i];

        return dSum;
    }
	
	double max() const
    {
        if(_iSize==0)
			return 0.; //not clean
		
		double dMax=_data[0];
        for(unsigned int i=1;i<_iSize;i++)
            if(_data[i]>dMax)
				dMax=_data[i];

        return dMax;
    }
	
	
    Matrix transpose() const // slow function!
    {
        Matrix out(_iColumns,_iRows);

        for(unsigned int r=0;r<_iRows;r++)
            for(unsigned int c=0;c<_iColumns;c++)
                out(c,r)=operator()(r,c);

        return out;
    }

    Matrix concat(const Matrix & b) // slow function!
    {
        assert(b.rows()==rows());

        Matrix mT(_iRows,_iColumns+b._iColumns);

        for(unsigned int r=0;r<_iRows;r++)
            for(unsigned int c=0;c<_iColumns;c++)
                mT(r,c)=operator()(r,c);

        for(unsigned int r=0;r<b.rows();r++)
            for(unsigned int c=0;c<b.columns();c++)
                mT(r,c+_iColumns)=b(r,c);

        return mT;
    }

    Matrix operator*(const Matrix& a) const  // slow function!
    {
        return Matrix(*this).operator*=(a);
    }


    Matrix row(unsigned int iRow)
    {
        assert(iRow<_iRows);

        return Matrix(_data+iRow*_iColumns,1,_iColumns);
    }

    
    const Matrix row(unsigned int iRow) const
    {
        assert(iRow<_iRows);

        return Matrix(_data+iRow*_iColumns,1,_iColumns);
    }


    Matrix row_sum() const
    {
        Matrix r(_iRows,1);

        for(unsigned int i=0;i<_iRows;i++)
            r(i)=row(i).sum();

        return r;
    }

    Matrix diag() const
    {
        Matrix r(_iRows,1);

        for(unsigned int i=0;i<_iRows;i++)
            r(i)=operator()(i,i);

        return r;
    }

    const Matrix without_last_row() const
    {
        assert(_iRows>0);
        return Matrix(_data,_iRows-1,_iColumns);
    }
    
    const Matrix without_last_column() const // slow function!
    {
        assert(_iColumns>0);

        Matrix m(_iRows,_iColumns-1);

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
    double* _data;
    bool _bDelete;
};

#endif
