/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#ifndef NetUtil_
#define NetUtil_

#include "Net.h"

#include <string>
using namespace std;

class Layer;
#include "Matrix.h"

namespace NetUtil {
string to_string(const Net* pNet);
Net* from_string(const string& s);

bool save(string sFileName,const Net* pNet);
bool load(string sFileName,Net* pNet);

};

#endif
