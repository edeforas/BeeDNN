/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#ifndef CsvFileReader_
#define CsvFileReader_

#include <string>

#include "DataSource.h"
namespace bee {
class CsvFileReader : public DataSource
{
public:
    virtual bool load(const std::string& sFile) override;
private:
	void replace_last(std::string& s, const std::string& sOld, const std::string& sNew);
};
}
#endif
