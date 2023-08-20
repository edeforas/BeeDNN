#include <iostream>
#include <cassert>
#include <chrono>

#include "Matrix.h"

using namespace std;
using namespace bee;

/////////////////////////////////////////////////////////////////////
// for testU only
inline bool is_near(double a, double b, double tolerancis = 1.e-4)
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
// https://stackoverflow.com/questions/18499971/efficient-4x4-matrix-multiplication-c-vs-assembly
void M4x4_SSE(const float* A, const float* B, float* C)
{
#if defined(_M_IX86_FP) || defined(_M_X64) ||defined(_M_AMD64)
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
			_mm_add_ps(	_mm_mul_ps(brod1, row1),	_mm_mul_ps(brod2, row2)),
			_mm_add_ps(	_mm_mul_ps(brod3, row3),	_mm_mul_ps(brod4, row4))
		);
		_mm_store_ps(&C[4 * i], row);
	}
#else
	assert(false);
#endif
}
////////////////////////////////////////////////////////
void GEMM_col_row_k_ptr(const MatrixFloat& a, const MatrixFloat& b, MatrixFloat& ab)
{
	ab.resize(a.rows(), b.cols());

	Index rows = ab.rows();
	Index cols = ab.cols();
	Index kMax = a.cols();

	vector<float> oneCol(kMax);

	for (Index c = 0; c < cols; c++)
	{
		//unstride stridded data (b column), cache friendly
		for (Index k = 0; k < kMax; k++)
			oneCol[k] = b(k, c);

		for (Index r = 0; r < rows; r++)
		{
			float temp = 0.f;
			const float * pA = a.row(r).data();
			const float * pB = oneCol.data();

			for (Index k = 0; k < kMax; k++)
				temp += (*pA++)*(*pB++);

			ab(r, c) = temp;
		}
	}
}
////////////////////////////////////////////////////////
void GEMM_row_col_k(const MatrixFloat& a, const MatrixFloat& b, MatrixFloat& ab)
{
	ab.setZero(a.rows(), b.cols());

	Index rows = ab.rows();
	Index cols = ab.cols();
	Index kMax= a.cols();

	for (Index r = 0; r < rows; r++)
		for (Index c = 0; c < cols; c++)
			for (Index k = 0; k < kMax; k++)
				ab(r, c) += a(r, k) * b(k, c);
}
////////////////////////////////////////////////////////
void GEMM_row_col_tiled(const MatrixFloat& a, const MatrixFloat& b, MatrixFloat& ab)
{
	ab.setZero(a.rows(), b.cols());

	//#define USE_SSE_TILESIZE
	
	const int tileSize = 16;
	Index rows = ab.rows();
	Index cols = ab.cols();
	Index kMax = a.cols();

	float a_block[tileSize* tileSize];
	float b_block[tileSize* tileSize];
	float ab_block[tileSize* tileSize];

	for (Index rb = 0; rb < rows; rb+= tileSize)
		for (Index cb = 0; cb < cols; cb+= tileSize)
			for(Index kb = 0; kb < kMax; kb+= tileSize)
			{
				//read the a block
				for (Index i = 0; i < tileSize; i++)
					for (Index j = 0; j < tileSize; j++)
						a_block[i * tileSize + j] = a(rb + i, kb + j);

				//  compute 4x4 MAC, use SSE if exist
				#if (defined(_M_IX86_FP) || defined(_M_X64) || defined(_M_AMD64)) && defined(USE_SSE_TILESIZE) 

				// read the b block
				for (Index i = 0; i < tileSize; i++)
					for (Index j = 0; j < tileSize; j++)
						b_block[i * tileSize + j] = b(kb + i, cb + j);
				
				M4x4_SSE(a_block, b_block, ab_block);
				
				#else

				// read the b block transposed
				for (Index i = 0; i < tileSize; i++)
					for (Index j = 0; j < tileSize; j++)
						b_block[j * tileSize + i] = b(kb + i, cb + j);

				// compute the blockSize*blockSize product, slower than SSE
				for (Index i = 0; i < tileSize; i++)
					for (Index j = 0; j < tileSize; j++)
					{
						float sum = 0.f;
						for (Index k = 0; k < tileSize; k++)
							sum += a_block[i * tileSize + k] * b_block[j* tileSize + k  ];
						ab_block[i * tileSize + j] = sum;
					}
				#endif
				// add to ab
				for (Index i = 0; i < tileSize; i++)
					for (Index j = 0; j < tileSize; j++)
						ab(rb + i, cb + j) += ab_block[i * tileSize + j];
			}
}
////////////////////////////////////////////////////////
void GEMM_row_k_col(const MatrixFloat& a, const MatrixFloat& b, MatrixFloat& ab)
{
	ab.setZero(a.rows(), b.cols());

	Index rows = ab.rows();
	Index cols = ab.cols();
	Index kMax = a.cols();

	for (Index r = 0; r < rows; r++)
		for (Index k = 0; k < kMax; k++)
			for (Index c = 0; c < cols; c++)
				ab(r, c) += a(r, k) * b(k, c);
}
////////////////////////////////////////////////////////
void GEMM_col_row_k(const MatrixFloat& a, const MatrixFloat& b, MatrixFloat& ab)
{
	ab.setZero(a.rows(), b.cols());

	Index rows = ab.rows();
	Index cols = ab.cols();
	Index kMax = a.cols();

	for (Index c = 0; c < cols; c++)
		for (Index r = 0; r < rows; r++)
			for (Index k = 0; k < kMax; k++)
				ab(r, c) += a(r, k) * b(k, c);
}
////////////////////////////////////////////////////////
void GEMM_col_k_row(const MatrixFloat& a, const MatrixFloat& b, MatrixFloat& ab)
{
	ab.setZero(a.rows(), b.cols());

	Index rows = ab.rows();
	Index cols = ab.cols();
	Index kMax = a.cols();

	for (Index c = 0; c < cols; c++)
		for (Index k = 0; k < kMax; k++)
			for (Index r = 0; r < rows; r++)
				ab(r, c) += a(r, k) * b(k, c);
}
////////////////////////////////////////////////////////
void GEMM_k_row_col(const MatrixFloat& a, const MatrixFloat& b, MatrixFloat& ab)
{
	ab.setZero(a.rows(), b.cols());

	Index rows = ab.rows();
	Index cols = ab.cols();
	Index kMax = a.cols();

	for (Index k = 0; k < kMax; k++)
		for (Index r = 0; r < rows; r++)
			for (Index c = 0; c < cols; c++)
				ab(r, c) += a(r, k) * b(k, c);
}
////////////////////////////////////////////////////////
void GEMM_k_col_row(const MatrixFloat& a, const MatrixFloat& b, MatrixFloat& ab)
{
	ab.setZero(a.rows(), b.cols());

	Index rows = ab.rows();
	Index cols = ab.cols();
	Index kMax = a.cols();

	for (Index k = 0; k < kMax; k++)
		for (Index c = 0; c < cols; c++)
			for (Index r = 0; r < rows; r++)
				ab(r, c) += a(r, k) * b(k, c);
}

////////////////////////////////////////////////////////
// from https://siboehm.com/articles/22/Fast-MMM-on-CPU#:~:text=Numpy%20can%20multiply%20two%201024x1024,for%20Basic%20Linear%20Algebra%20Subprograms.
void matmulImplRowColParallelInnerTiling(int rows, int columns, int inners, const float *left, const float *right, float *result)
{
	int iThreadTileSize = 256;
	int tileSize = 4;

	#pragma omp parallel for shared(result, left, right) default(none) num_threads(8)
	for (int rowTile = 0; rowTile < rows; rowTile += iThreadTileSize)
	{		  
		int maxRow = std::min(rowTile + iThreadTileSize, rows);
		for (int columnTile = 0; columnTile < columns; columnTile += iThreadTileSize)
		{		
			int maxCol = std::min(columnTile + iThreadTileSize, columns);
			for (int innerTile = 0; innerTile < inners; innerTile += tileSize)
			{		
				int innerTileEnd = std::min(inners, innerTile + tileSize); 
				for (int row = rowTile; row < maxRow; row++)
				{
					for (int inner = innerTile; inner < innerTileEnd; inner++)
					{
						for (int col = columnTile; col < maxCol; col++)
						{
							result[row * columns + col] +=  left[row * inners + inner] * right[inner * columns + col];
						}
					}
				}
			}
		}
	}
}
void GEMM_tiled_omp(const MatrixFloat& a, const MatrixFloat& b, MatrixFloat& ab)
{
	ab.setZero(a.rows(), b.cols());
	matmulImplRowColParallelInnerTiling((int)a.rows(), (int)b.rows(), (int)a.cols(), a.data(), b.data(), ab.data());
}
////////////////////////////////////////////////////////
void test_GEMM_order()
{
	using namespace std;
	std::cout << endl << "GEMM order test:" << endl;
	
	chrono::steady_clock::time_point start, end;

	// for this test, all sizes are multiples of 4
	for (int sz = 16; sz <= 1024; sz *= 2)
	{
		MatrixFloat m1;
		m1.setRandom(sz, sz);

		MatrixFloat m2;
		m2.setRandom(sz, sz);

		start = chrono::steady_clock::now();
		MatrixFloat m3optim = m1 * m2;
		end = chrono::steady_clock::now();
		long long deltaOptim = chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

		start = chrono::steady_clock::now();
		MatrixFloat m3Naive1;
		GEMM_row_col_k(m1, m2, m3Naive1);
		end = chrono::steady_clock::now();
		long long delta_row_col_k = chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		float fErrorNaive1 = (m3optim - m3Naive1).cwiseAbs().maxCoeff();
		test(is_near(fErrorNaive1, 0.));

		start = chrono::steady_clock::now();
		MatrixFloat m3Naive2;
		GEMM_row_k_col(m1, m2, m3Naive2);
		end = chrono::steady_clock::now();
		long long delta_row_k_col = chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		float fErrorNaive2 = (m3optim - m3Naive2).cwiseAbs().maxCoeff();
		test(is_near(fErrorNaive2, 0.));

		start = chrono::steady_clock::now();
		MatrixFloat m3Naive3;
		GEMM_col_row_k(m1, m2, m3Naive3);
		end = chrono::steady_clock::now();
		long long delta_col_row_k = chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		float fErrorNaive3 = (m3optim - m3Naive3).cwiseAbs().maxCoeff();
		test(is_near(fErrorNaive3, 0.));

		start = chrono::steady_clock::now();
		MatrixFloat m3Naive4;
		GEMM_col_k_row(m1, m2, m3Naive4);
		end = chrono::steady_clock::now();
		long long delta_col_k_row = chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		float fErrorNaive4 = (m3optim - m3Naive4).cwiseAbs().maxCoeff();
		test(is_near(fErrorNaive4, 0.));

		start = chrono::steady_clock::now();
		MatrixFloat m3Naive5;
		GEMM_k_row_col(m1, m2, m3Naive5);
		end = chrono::steady_clock::now();
		long long delta_k_row_col = chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		float fErrorNaive5 = (m3optim - m3Naive5).cwiseAbs().maxCoeff();
		test(is_near(fErrorNaive5, 0.));

		start = chrono::steady_clock::now();
		MatrixFloat m3Naive6;
		GEMM_k_col_row(m1, m2, m3Naive6);
		end = chrono::steady_clock::now();
		long long delta_k_col_row = chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		float fErrorNaive6 = (m3optim - m3Naive6).cwiseAbs().maxCoeff();
		test(is_near(fErrorNaive6, 0.));

		start = chrono::steady_clock::now();
		MatrixFloat m3Naive7;
		GEMM_col_row_k_ptr(m1, m2, m3Naive7);
		end = chrono::steady_clock::now();
		long long delta_col_row_k_ptr = chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		float fErrorNaive7 = (m3optim - m3Naive7).cwiseAbs().maxCoeff();
		test(is_near(fErrorNaive7, 0.));

		start = chrono::steady_clock::now();
		MatrixFloat m3Naive8;
		GEMM_row_col_tiled(m1, m2, m3Naive8);
		end = chrono::steady_clock::now();
		long long delta_row_col_tiled = chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		float fErrorNaive8 = (m3optim - m3Naive8).cwiseAbs().maxCoeff();
		test(is_near(fErrorNaive8, 0.));

		start = chrono::steady_clock::now();
		MatrixFloat m3Naive9;
		GEMM_tiled_omp(m1, m2, m3Naive9);
		end = chrono::steady_clock::now();
		long long delta_tiled_omp = chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		float fErrorNaive9 = (m3optim - m3Naive9).cwiseAbs().maxCoeff();
		test(is_near(fErrorNaive9, 0.));

		cout << "size:" << sz << "  Optim/RCK/RKC/CRK/CKR/KRC/KCR/CRKptr/Tiled/TiledOMP time: "
			<< deltaOptim << "/"
			<< delta_row_col_k << "/"
			<< delta_row_k_col << "/"
			<< delta_col_row_k << "/"
			<< delta_col_k_row << "/"
			<< delta_k_row_col << "/"
			<< delta_k_col_row << "/"
			<< delta_col_row_k_ptr << "/"
			<< delta_row_col_tiled << "/"
			<< delta_tiled_omp << "/"
			<< endl;
	}
}
////////////////////////////////////////////////////////

int main()
{
	test_GEMM_order();

    std::cout << "Tests finished." << std::endl;
    return 0;
}
////////////////////////////////////////////////////////