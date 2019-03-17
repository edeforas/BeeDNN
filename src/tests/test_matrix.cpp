#include <iostream>
#include <cassert>
using namespace std;

#include "Matrix.h"

////////////////////////////////////////////////////////
void disp(const MatrixFloat& m)
{
    cout << "rows=" << m.rows() << " columns=" << m.cols() << endl;
    for(unsigned int r=0;r<m.rows();r++)
    {
        for(unsigned int c=0;c<m.cols();c++)
            cout << m(r,c) << " ";
        cout << endl;
    }
}
////////////////////////////////////////////////////////
void check_fromRawBuffer()
{
    //check fromRawBuffer() is not copying the data, i.e. is a real view
    float c[5]={ 0, 1 , 2 , 3 , 4 };
    MatrixFloatView mC=fromRawBuffer(c,5,1);
    c[0]=333;
    assert( (mC(0,0)==333) && "fromRawBuffer() must not copy the data" );

    // do we accept const?
    const float d[5]={ 0, 1 , 2 , 3 , 4 };
    const MatrixFloatView mD=fromRawBuffer(d,5,1);
    (void)mD;
}
////////////////////////////////////////////////////////
int main()
{
    float a[]={ 4 , 5 , 6, 7 };
    float b[]={ 2 , 3 };

    const MatrixFloat mA=fromRawBuffer(a,2,2);
    const MatrixFloat mB=fromRawBuffer(b,2,1);

    const MatrixFloat mAT=mA.transpose();
	cout << "Transposed Matrix:" << endl;
    disp(mAT);

    MatrixFloat prod=mA*mB;
	cout << "Product Matrix:" << endl;
    disp(prod);

	MatrixFloat mD=mA.diagonal();
	cout << "Diagonal Matrix as vector:" << endl;
	disp(mD);

    MatrixFloat mS=rowWiseSum(mA);
    cout << "RowWiseSum:" << endl;
    disp(mS);

    check_fromRawBuffer();

    cout << "Tests ok!" << endl;
    return 0;
}
