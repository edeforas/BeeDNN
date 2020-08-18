#ifndef DataSourceSelector_
#define DataSourceSelector_

#include <string>
using namespace std;

#include "Matrix.h"

class DataSourceSelector
{
public:
    DataSourceSelector();
    virtual ~DataSourceSelector();

    void clear();

    void write(string& s) const;
    void read(const string& s);

    const string name() const;

    virtual bool load(const string & sName);

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
	string _sName;

private:
	bool load_mnist();
	bool load_mini_mnist();
	bool load_cifar10();
	bool load_textfile();	
};

#endif
