#ifndef DataSource_
#define DataSource_

#include <string>
using namespace std;

#include "Matrix.h"

class DataSource
{
public:
    DataSource();
    virtual ~DataSource();

	virtual bool load(const string & sName) = 0;

    const MatrixFloat& train_data() const;
    const MatrixFloat& train_truth() const;

    const MatrixFloat& test_data() const;
    const MatrixFloat& test_truth() const;

    bool has_data() const;
    bool has_train_data() const;
    bool has_test_data() const;

    int data_size() const;
    int annotation_cols() const;

protected:
    MatrixFloat _mTrainData;
    MatrixFloat _mTrainTruth;
    MatrixFloat _mTestData;
    MatrixFloat _mTestTruth;

    bool _bHasTestData;
    bool _bHasTrainData;
};

#endif
