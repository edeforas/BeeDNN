#ifndef MNISTReader_
#define MNISTReader_

#include "Matrix.h"

#include <string>
using namespace std;

class MNISTReader
{
public:
    bool read_from_folder(const string& sFolder,Matrix& mRefImages,Matrix& mRefLabels,Matrix& mTestImages,Matrix& mTestLabels);
    bool read_matrix(string sName,Matrix& m);
};

#endif
