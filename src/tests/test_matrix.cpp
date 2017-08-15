#include <iostream>
using namespace std;

#include "Matrix.h"

void disp(const Matrix& m)
{
    cout << "rows=" << m.rows() << " columns=" << m.columns() << endl;
    for(unsigned int r=0;r<m.rows();r++)
    {
        for(unsigned int c=0;c<m.columns();c++)
            cout << m(r,c) << " ";
        cout << endl;
    }
}


int main()
{
    double a[]={ 4 , 5 , 6, 7 };
    double b[]={ 2, 3};

    const Matrix mA(a,2,2);
    const Matrix mB(b,2,1);

    const Matrix mAT=mA.transpose();
    disp(mAT);

    Matrix prod=mA*mB;

    disp(prod);

    return 0;
}
