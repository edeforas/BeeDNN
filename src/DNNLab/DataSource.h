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

    void write(string& s);
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
    void load_mnist();
	void load_mini_mnist();
    void load_function();
    void load_textfile();
    void load_and();
    void load_xor();
    void load_fisher();
	float get_function_val(float x);

    MatrixFloat _mTrainData;
    MatrixFloat _mTrainTruth;
    MatrixFloat _mTestData;
    MatrixFloat _mTestTruth;

    bool _bHasTestData;
    bool _bHasTrainData;

    string _sName;
};

#endif
