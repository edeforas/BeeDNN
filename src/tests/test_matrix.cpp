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
    float b[]={ 2, 3};

    const MatrixFloat mA(a,2,2);
    const MatrixFloat mB(b,2,1);

    const MatrixFloat mAT=mA.transpose();
    disp(mAT);

    MatrixFloat prod=mA*mB;

    disp(prod);

    return 0;
}
