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

    void clear();

    void write(string& s) const;
    void read(const string& s);

    const string name() const;

    void load(const string & sName);

    const MatrixFloat& train_data() const;
    const MatrixFloat& train_truth() const;

    const MatrixFloat& test_data() const;
    const MatrixFloat& test_truth() const;

    bool has_data() const;
    bool has_train_data() const;
    bool has_test_data() const;

    int data_size() const;
    int annotation_cols() const;

private:
    bool load_mnist();
    bool load_mini_mnist();
    bool load_cifar10();
    bool load_textfile();

    void load_function();
    void load_and();
    void load_xor();

    float get_function_val(float x) const;

    MatrixFloat _mTrainData;
    MatrixFloat _mTrainTruth;
    MatrixFloat _mTestData;
    MatrixFloat _mTestTruth;

    bool _bHasTestData;
    bool _bHasTrainData;

    string _sName;
};

#endif