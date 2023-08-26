/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#pragma once

#include <string>
namespace beednn {
class Net;
class NetTrain;

namespace NetUtil {

    void save(const std::string& sFile, const beednn::Net& model, const beednn::NetTrain& trainParams);

    //void read(const string& s,Net& net);
    //void write(const NetTrain& train,string& s);
    //void read(const string& s,NetTrain& train);
    //string find_key(string s,string sKey);
    //void split(string s, vector<string>& vsItems, char cDelimiter=' ');
}
}
