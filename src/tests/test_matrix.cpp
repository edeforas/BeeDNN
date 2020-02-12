#include <iostream>
#include <cassert>
#include <chrono>
using namespace std;

#include "Matrix.h"

////////////////////////////////////////////////////////
void disp(const MatrixFloat& m)
{
    cout << "rows=" << m.rows() << " columns=" << m.cols() << endl;
    for( int r=0;r<m.rows();r++)
    {
        for( int c=0;c<m.cols();c++)
            cout << m(r,c) << " ";
        cout << endl;
    }
}
////////////////////////////////////////////////////////
void elementary_tests()
{
	float a[] = { 4 , 5 , 6, 7 };
	float b[] = { 2 , 3 };

	const MatrixFloat mA = fromRawBuffer(a, 2, 2);
	const MatrixFloat mB = fromRawBuffer(b, 2, 1);

	const MatrixFloat mAT = mA.transpose();
	cout << "Transposed Matrix:" << endl;
	disp(mAT);

	MatrixFloat prod = mA * mB;
	cout << "Product Matrix:" << endl;
	disp(prod);

	MatrixFloat mD = mA.diagonal();
	cout << "Diagonal Matrix as vector:" << endl;
	disp(mD);

	MatrixFloat mS = rowWiseSum(mA);
	cout << "RowWiseSum:" << endl;
	disp(mS);
}
////////////////////////////////////////////////////////////
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
void test_bernoulli()
{
	MatrixFloat m(1000, 1000);
	chrono::steady_clock::time_point start = chrono::steady_clock::now();
	for(int i=1;i<10;i++)
		setBernoulli(m, 0.5f);

	chrono::steady_clock::time_point end = chrono::steady_clock::now();
	auto delta = chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	cout << "Bernoulli Time elapsed: " << delta << " ms" << endl;
}
////////////////////////////////////////////////////////
int main()
{
	elementary_tests();
    check_fromRawBuffer();
	test_bernoulli();

    cout << "Tests finished." << endl;
    return 0;
}
////////////////////////////////////////////////////////