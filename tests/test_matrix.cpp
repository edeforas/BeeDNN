#include <iostream>
#include <cassert>
#include <chrono>

#include "Matrix.h"
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

	std::cout << "Test failed: " << sMessage << std::endl;
	exit(-1);
}
////////////////////////////////////////////////////////
void disp(const MatrixFloat& m)
{
    std::cout << "rows=" << m.rows() << " columns=" << m.cols() << std::endl;
    for( int r=0;r<m.rows();r++)
    {
        for( int c=0;c<m.cols();c++)
            std::cout << m(r,c) << " ";
        std::cout << std::endl;
    }
}
////////////////////////////////////////////////////////
void test_elementary()
{
	float a[] = { 4 , 5 , 6, 7 };
	float b[] = { 2 , 3 };

	const MatrixFloat mA = fromRawBuffer(a, 2, 2);
	const MatrixFloat mB = fromRawBuffer(b, 2, 1);

	const MatrixFloat mAT = mA.transpose();
	std::cout << "Transposed Matrix:" << std::endl;
	disp(mAT);

	MatrixFloat prod = mA * mB;
	std::cout << "Product Matrix:" << std::endl;
	disp(prod);

	MatrixFloat mD = mA.diagonal();
	std::cout << "Diagonal Matrix as vector:" << std::endl;
	disp(mD);

	MatrixFloat mS = rowWiseSum(mA);
	std::cout << "RowWiseSum:" << std::endl;
	disp(mS);
}
////////////////////////////////////////////////////////////
void test_matrixView()
{
	std::cout << "check_matrixView:" << std::endl;

    //check fromRawBuffer() is not copying the data, i.e. is a real view
    float c[5]={ 0, 1 , 2 , 3 , 4 };
    MatrixFloatView mC=fromRawBuffer(c,5,1);
    c[0]=333;
    assert( (mC(0,0)==333) && "fromRawBuffer() must not copy the data" );

    // do we accept const?
	float d[5]={ 0, 1 , 2 , 3 , 4 };
    const MatrixFloatView mD=fromRawBufferConst(d,5,1);
    (void)mD;

	//check matrix view is not copying the data, i.e. is a real view
	MatrixFloat mf(2, 2);
	mf(0,0) = 0; mf(1) = 1; mf(2) = 2; mf(3) = 3;
	MatrixFloatView mV= createView(mf);
	MatrixFloat mV2 = createView(mf);
	MatrixFloat mV3 = createView(mV2); //view on view

	mf(2) = 333;
	test(is_near(mV(2), 333), "mV fromRawBuffer() must not copy the data");
	test(is_near(mV2(2),2),"mV2 fromRawBuffer() must copy the data");
	test(is_near(mV3(2), 2), "mV3 fromRawBuffer() must copy the data");

	std::cout << "check_matrixView finished" << std::endl;
}
////////////////////////////////////////////////////////
void test_bernoulli()
{
	std::cout << "test_bernoulli:" << std::endl;
	MatrixFloat m(1000, 1000);

	std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
	for (int itest = 1; itest < 10; itest++)
	{
		std::bernoulli_distribution dis(0.3f);
		for (Index i = 0; i < m.size(); i++)
			m(i) = (float)(dis(randomEngine())); //slow 
	}

	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
	auto delta = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	std::cout << "Slow Bernoulli Time elapsed: " << delta << " ms. Mean= " << m.mean() << std::endl;
	test(is_near(m.mean(),0.3f, 0.001),"Mean must be near 0.3f");

	start = std::chrono::steady_clock::now();
	for(int i=1;i<10;i++)
		setQuickBernoulli(m, 0.3f);

	 end = std::chrono::steady_clock::now();
	 delta = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	std::cout << "Quick Bernoulli Time elapsed: " << delta << " ms. Mean= " << m.mean() << std::endl;
	test(is_near(m.mean(), 0.3f, 0.001), "Mean must be near 0.3f");
}
////////////////////////////////////////////////////////
void test_hyperbolic()
{
	std::cout << "test_tanh:" << std::endl;

	MatrixFloat m1(1000, 1000), m2(1000, 1000), m(1000, 1000);
	m.setRandom();

	{
		auto start = std::chrono::steady_clock::now();
		for (int i = 0; i < 100; i++)
			m1 = m.array().tanh();
		auto end = std::chrono::steady_clock::now();
		auto delta = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		std::cout << "Tanh Time elapsed: " << delta << " ms. Mean= " << m1.mean() << std::endl;
	}
	
	{	
		auto start = std::chrono::steady_clock::now();
		for (int i = 0; i < 100; i++)
			for (int j = 0; j < m.size(); j++)
				m2(j) = tanh(m(j));
		auto end = std::chrono::steady_clock::now();
		auto delta = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		std::cout << "Tanh Time elapsed: " << delta << " ms. Mean= " << m2.mean() << std::endl;
	}
}
////////////////////////////////////////////////////////
int main()
{
	test_elementary();
	test_matrixView();
	test_bernoulli();
	test_hyperbolic();

    std::cout << "Tests finished." << std::endl;
    return 0;
}
////////////////////////////////////////////////////////