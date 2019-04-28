#include <iostream>
#include <chrono>
#include <thread>
using namespace std;

#include "Matrix.h"

// this test help to measure the matrix product time, for later architecture optimizations
// options are with or without OpenMP (option under vs2017), with or without eigen using the preprocessor define USE_EIGEN

//////////////////////////////////////////////////////////////////////////
int main()
{
	for (int i = 64; i <= 8192; i *= 2)
	{
		MatrixFloat A; A.setRandom(i, i);
		MatrixFloat B; B.setRandom(i, i);
		MatrixFloat C; C.setRandom(i, i);
		MatrixFloat D; D.setRandom(i, i);

		std::chrono::time_point<std::chrono::system_clock> start, end;
		start = std::chrono::system_clock::now();

		D = A * B + C;

		end = std::chrono::system_clock::now();
		int elapsed_ms = (int)(std::chrono::duration_cast<std::chrono::milliseconds> 	(end - start).count());
		std::cout << "Size=" << i << "x" << i << " time(ms)/megapixels=" << elapsed_ms*1000000./(i*i) << "ms" << " total_time(ms)=" << elapsed_ms << "ms" << endl;
	}

	return 0;
}
