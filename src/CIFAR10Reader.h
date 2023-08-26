/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#ifndef CIFAR10Reader_
#define CIFAR10Reader_

#include "Matrix.h"

#include <string>

#include "DataSource.h"
namespace beednn {
class CIFAR10Reader : public DataSource
{
public:
    virtual bool load(const std::string& sFolder) override;

private:
    bool read_from_folder(const std::string& sFolder,MatrixFloat& mRefImages,MatrixFloat& mRefLabels,MatrixFloat& mTestImages,MatrixFloat& mTestLabels);
    bool read_batch(std::string sName,MatrixFloat& mData,MatrixFloat& mTruth);
};
}
#endif
