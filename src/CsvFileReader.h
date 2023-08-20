/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#ifndef CsvFileReader_
#define CsvFileReader_

#include <string>
using namespace std;

#include "DataSource.h"
namespace bee {
class CsvFileReader : public DataSource
{
public:
    virtual bool load(const string& sFile) override;
private:
	void replace_last(string& s, const string& sOld, const string& sNew);
};
}
#endif
