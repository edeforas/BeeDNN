#include <iostream>
#include <cassert>
#include <chrono>
using namespace std;

#include "Matrix.h"

//from https://bulyaki.wordpress.com/2018/05/14/single-and-double-precision-4x4-and-8x8-block-matrix-products-using-sse2-avx-and-avx512-intrinsics/
struct Mat44f
{
	float value[4][4];
};
void MultiplyAndAddMM4x4F(Mat44f & result, const Mat44f & MatrixA, const Mat44f & MatrixB)
{
	__m128 rightRow[4];
	__m128 resultRow[4];
	for (int i = 0; i < 4; ++i)
	{
		rightRow[i] = _mm_loadu_ps((const float *)MatrixB.value[i]);
		resultRow[i] = _mm_loadu_ps((const float *)result.value[i]);
	}
	for (int i = 0; i < 4; ++i)
	{
		resultRow[i] = _mm_add_ps(resultRow[i], _mm_mul_ps(rightRow[0], _mm_set1_ps(MatrixA.value[i][0])));
		resultRow[i] = _mm_add_ps(resultRow[i], _mm_mul_ps(rightRow[1], _mm_set1_ps(MatrixA.value[i][1])));
		resultRow[i] = _mm_add_ps(resultRow[i], _mm_mul_ps(rightRow[2], _mm_set1_ps(MatrixA.value[i][2])));
		resultRow[i] = _mm_add_ps(resultRow[i], _mm_mul_ps(rightRow[3], _mm_set1_ps(MatrixA.value[i][3])));

		_mm_storeu_ps((float *)result.value[i], resultRow[i]);
	}
}
////////////////////////////////////////////////////////
// https://stackoverflow.com/questions/18499971/efficient-4x4-matrix-multiplication-c-vs-assembly
void M4x4_SSE(const float *A,const float *B, float *C)
{
	__m128 row1 = _mm_load_ps(&B[0]);
	__m128 row2 = _mm_load_ps(&B[4]);
	__m128 row3 = _mm_load_ps(&B[8]);
	__m128 row4 = _mm_load_ps(&B[12]);
	for (int i = 0; i < 4; i++) {
		__m128 brod1 = _mm_set1_ps(A[4 * i + 0]);
		__m128 brod2 = _mm_set1_ps(A[4 * i + 1]);
		__m128 brod3 = _mm_set1_ps(A[4 * i + 2]);
		__m128 brod4 = _mm_set1_ps(A[4 * i + 3]);
		__m128 row = _mm_add_ps(
			_mm_add_ps(
				_mm_mul_ps(brod1, row1),
				_mm_mul_ps(brod2, row2)),
			_mm_add_ps(
				_mm_mul_ps(brod3, row3),
				_mm_mul_ps(brod4, row4)));
		_mm_store_ps(&C[4 * i], row);
	}
}
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
	test(is_near(mV2(2),2),"mV2 fromRawBuffer() must copy the data");
	test(is_near(mV3(2), 2), "mV3 fromRawBuffer() must copy the data");

	cout << "check_matrixView finished" << endl;
}
////////////////////////////////////////////////////////
void test_bernoulli()
{
	cout << "test_bernoulli:" << endl;
	MatrixFloat m(1000, 1000);

	chrono::steady_clock::time_point start = chrono::steady_clock::now();
	for (int itest = 1; itest < 10; itest++)
	{
		bernoulli_distribution dis(0.3f);
		for (Index i = 0; i < m.size(); i++)
			m(i) = (float)(dis(randomEngine())); //slow 
	}

	chrono::steady_clock::time_point end = chrono::steady_clock::now();
	auto delta = chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	cout << "Slow Bernoulli Time elapsed: " << delta << " ms. Mean= " << m.mean() << endl;
	test(is_near(m.mean(),0.3f, 0.001),"Mean must be near 0.3f");

	start = chrono::steady_clock::now();
	for(int i=1;i<10;i++)
		setQuickBernoulli(m, 0.3f);

	 end = chrono::steady_clock::now();
	 delta = chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	cout << "Quick Bernoulli Time elapsed: " << delta << " ms. Mean= " << m.mean() << endl;
	test(is_near(m.mean(), 0.3f, 0.001), "Mean must be near 0.3f");
}
////////////////////////////////////////////////////////
void GEMM_naive(const MatrixFloat& a, const MatrixFloat& b, MatrixFloat& ab)
{
	ab.resize(a.rows(), b.cols());

	Index rows = ab.rows();
	Index cols = ab.cols();
	Index inCols = a.cols();
	
	for (Index r = 0; r < rows; r++)
	{
		for (Index c = 0; c < cols; c++)
		{
			float temp = 0.f;

			for (Index k = 0; k < inCols; k++)
				temp += a(r, k)*b(k, c);

			ab(r, c) = temp;
		}
	}
}
////////////////////////////////////////////////////////
void GEMM_naive2(const MatrixFloat& a, const MatrixFloat& b, MatrixFloat& ab)
{
	ab.resize(a.rows(), b.cols());

	Index rows = ab.rows();
	Index cols = ab.cols();
	Index inDepth = a.cols();

	vector<float> oneCol(inDepth);

	for (Index c = 0; c < cols; c++)
	{
		//unstripe data
		for (Index r = 0; r < inDepth; r++)
			oneCol[r] = b(r, c);

		for (Index r = 0; r < rows; r++)
		{
			float temp = 0.f;
			const float * pA = a.row(r).data();
			const float * pB = oneCol.data();

			for (Index k = 0; k < inDepth; k++)
				temp += (*pA++)*(*pB++);

			ab(r, c) = temp;
		}
	}
}
////////////////////////////////////////////////////////
void test_GEMM()
{
	cout << endl << "GEMM test:" << endl;

	chrono::steady_clock::time_point start, end;

	for (int sz = 1; sz <= 2048; sz*=2)
	{
		MatrixFloat m1(sz, sz);
		m1.setRandom();

		MatrixFloat m2(sz, sz);
		m2.setRandom();

		start = chrono::steady_clock::now();
		MatrixFloat m3optim= m1 * m2;
		end = chrono::steady_clock::now();
		long long deltaOptim = chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

		start = chrono::steady_clock::now();
		MatrixFloat m3Naive;
		GEMM_naive(m1, m2, m3Naive);
		end = chrono::steady_clock::now();
		long long deltaNaive = chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		
		start = chrono::steady_clock::now();
		MatrixFloat m3Naive2;
		GEMM_naive2(m1, m2, m3Naive2);
		end = chrono::steady_clock::now();
		long long deltaNaive2 = chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

		float fErrorNaive = (m3optim - m3Naive).cwiseAbs().maxCoeff();
		float fErrorNaive2 = (m3optim - m3Naive2).cwiseAbs().maxCoeff();

		cout << "Optim/Naive/Naive2 size:" << sz << " time: " << deltaOptim << "/" << deltaNaive << "/" << deltaNaive2 << " errNaive: " << fErrorNaive << " errNaive2: " << fErrorNaive2 << endl;
	}
}
////////////////////////////////////////////////////////
void test_sse()
{
	Mat44f m1,m2,mr;
	MatrixFloat mf1(4, 4), mf2(4, 4),mfr;
	float mt1[16], mt2[16], mt3[16];
	for(int i=0;i<4;i++)
		for (int j = 0; j < 4; j++)
		{
			float a = (float)(i + j * 2 + 1);
			float b = (float)(i - j);

			m1.value[i][j] = a;
			m2.value[i][j] = b;
			mr.value[i][j] = 0;

			mf1(i, j) = a;
			mf2(i, j) = b;

			mt1[i * 4 + j] = a;
			mt2[i * 4 + j] = b;
			mt3[i * 4 + j] = 0;
		}

	mfr = mf1 * mf2; //eigen mmult
	for (int i = 0; i < 4; i++)
		for (int j = 0; j < 4; j++)
			cout << mfr(i, j) << " ";

	cout << endl;

	MultiplyAndAddMM4x4F(mr, m1, m2);
	for (int i = 0; i < 4; i++)
		for (int j = 0; j < 4; j++)
			cout << mr.value[i][j] << " ";

	cout << endl;
	M4x4_SSE(mt1, mt2, mt3);
	for (int i = 0; i < 16; i++)
		cout << mt3[i] << " ";
}
////////////////////////////////////////////////////////
int main()
{
	elementary_tests();
	check_matrixView();
	test_bernoulli();
	test_GEMM();
	test_sse();

    cout << "Tests finished." << endl;
    return 0;
}
////////////////////////////////////////////////////////