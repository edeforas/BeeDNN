#ifndef DataSource_
#define DataSource_

#include <string>

#include "Matrix.h"
namespace bee {
class DataSource
{
public:
    DataSource();
    virtual ~DataSource();

	virtual bool load(const std::string & sName) = 0;

    const MatrixFloat& train_data() const;
    const MatrixFloat& train_truth() const;

    const MatrixFloat& validation_data() const;
    const MatrixFloat& validation_truth() const;

    bool has_data() const;
    bool has_train_data() const;
    bool has_validation_data() const;

    int data_size() const;
    int annotation_cols() const;

protected:
    MatrixFloat _mTrainData;
    MatrixFloat _mTrainTruth;
    MatrixFloat _mValData;
    MatrixFloat _mValTruth;

    bool _bHasTrainData;
	bool _bHasValidationData;
};
}
#endif
