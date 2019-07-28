/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#ifndef NetUtil_
#define NetUtil_

class Net;
class NetTrain;

#include <string>
using namespace std;

class Layer;
#include "Matrix.h"

namespace NetUtil {

void write(const Net& net,string& s);
void read(const string& s,Net& net);

void write(const NetTrain& train,string& s);
void read(const string& s,NetTrain& train);

bool save(string sFileName,const Net& net);
bool load(string sFileName,Net* pNet);

string find_key(string s,string sKey);

};

#endif
