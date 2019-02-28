#include <iostream>
using namespace std;

#include "Matrix.h"

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

int main()
{
    float a[]={ 4 , 5 , 6, 7 };
    float b[]={ 2 , 3 };

    const MatrixFloat mA=from_raw_buffer(a,2,2);
    const MatrixFloat mB=from_raw_buffer(b,2,1);

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




    return 0;
}
