#ifndef DataSource_
#define DataSource_

#include <string>
using namespace std;

#include "Matrix.h"

class DataSource
{
public:
    DataSource();
    ~DataSource();

    void load_mnist();
    void load_function(string sFunction,float fMin, float fMax, int iNbPoints);
    void load_textfile();
    void load_and();
    void load_xor();

    const MatrixFloat& train_data() const;
    const MatrixFloat& train_annotation() const;

    const MatrixFloat& test_data() const;
    const MatrixFloat& test_annotation() const;

    bool has_data() const;
    bool has_train_data() const;
    bool has_test_data() const;

    int data_cols() const;
    int annotation_cols() const;

private:
    MatrixFloat _mTrainData;
    MatrixFloat _mTrainAnnotation;
    MatrixFloat _mTestData;
    MatrixFloat _mTestAnnotation;

    bool _bHasTestData;
    bool _bHasTrainData;

    string _sLastLoaded;
};

#endif
