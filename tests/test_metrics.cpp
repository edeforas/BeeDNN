#include <iostream>
#include <cassert>

#include "Matrix.h"
#include "Metrics.h"

using namespace std;
using namespace bee;
/////////////////////////////////////////////////////////////////////
// for testU only
inline bool is_near(double a, double b, double tolerancis = 1.e-10)
{
	return fabs(a - b) < tolerancis;
}
void test(bool bTest, const string& sMessage = "")
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
void test_accuracy()
{
	cout << "test accuracy:" << endl;
	
	float a[] = { 0,1,2,3};
	float b[] = { 0,1,2,3 };

	const MatrixFloat mA = fromRawBuffer(a, 3, 1);
	const MatrixFloat mB = fromRawBuffer(b, 3, 1);

	Metrics metrics;
	metrics.compute(mA,mB);
	
	test(is_near(metrics.accuracy(), 100.f), "Accuracy must = 100%");
	test(is_near(metrics.balanced_accuracy(),100.f), "Balanced Accuracy must = 100%");
	//test(cr.mConfMat==MatrixFloat::Identity(3),"sqldfl")
}
////////////////////////////////////////////////////////
int main()
{
	test_accuracy();

    cout << "Tests finished." << endl;
    return 0;
}
////////////////////////////////////////////////////////