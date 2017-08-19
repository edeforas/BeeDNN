#ifndef MNISTReader_
#define MNISTReader_

#include "Matrix.h"

#include <string>
using namespace std;

class MNISTReader
{
public:
    bool read_from_folder(const string& sFolder,MatrixFloat& mRefImages,MatrixFloat& mRefLabels,MatrixFloat& mTestImages,MatrixFloat& mTestLabels);
    bool read_Matrix(string sName,MatrixFloat& m);

private:
    void swap_int(unsigned int &i);
};

#endif
