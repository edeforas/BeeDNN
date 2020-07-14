#include <iostream>
#include <cassert>
#include <chrono>
using namespace std;

#include "Matrix.h"

/////////////////////////////////////////////////////////////////////
// for testU only
inline bool is_near(double a, double b, double tolerancis = 1.e-10)
{
	return fabs(a - b) < tolerancis;
}
void test(bool bTest, string sMessage = "")
{
	if (bTest) return;

	cout << "Test failed: " << sMessage << endl;
	exit(-1);
}
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
void check_matrixView()
{
	cout << "check_matrixView:" << endl;

    //check fromRawBuffer() is not copying the data, i.e. is a real view
    float c[5]={ 0, 1 , 2 , 3 , 4 };
    MatrixFloatView mC=fromRawBuffer(c,5,1);
    c[0]=333;
    assert( (mC(0,0)==333) && "fromRawBuffer() must not copy the data" );

    // do we accept const?
    const float d[5]={ 0, 1 , 2 , 3 , 4 };
    const MatrixFloatView mD=fromRawBuffer(d,5,1);
    (void)mD;

	//check matrix view is not copying the data, i.e. is a real view
	MatrixFloat mf(2, 2);
	mf(0,0) = 0; mf(1) = 1; mf(2) = 2; mf(3) = 3;
	MatrixFloatView mV= createView(mf);
	MatrixFloat mV2 = createView(mf);
	MatrixFloat mV3 = createView(mV2); //view on view

	mf(2) = 333;
	test(is_near(mV(2), 333), "mV fromRawBuffer() must not copy the data");
	test(is_near(mV2(2),333),"mV2 fromRawBuffer() must not copy the data");
	test(is_near(mV3(2), 333), "mV3 fromRawBuffer() must not copy the data");

	cout << "check_matrixView finished" << endl;
}
////////////////////////////////////////////////////////
void test_bernoulli()
{
	cout << "test_bernoulli:" << endl;

	MatrixFloat m(1000, 1000);
	chrono::steady_clock::time_point start = chrono::steady_clock::now();
	for(int i=1;i<10;i++)
		setQuickBernoulli(m, 0.3f);

	chrono::steady_clock::time_point end = chrono::steady_clock::now();
	auto delta = chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	cout << "Bernoulli Time elapsed: " << delta << " ms. Mean= " << m.mean() << endl;
}
////////////////////////////////////////////////////////
int main()
{
	elementary_tests();
	check_matrixView();
	test_bernoulli();

    cout << "Tests finished." << endl;
    return 0;
}
////////////////////////////////////////////////////////