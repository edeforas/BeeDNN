/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#ifndef JsonFile_
#define JsonFile_

#include <string>
#include <vector>
#include <map>
using namespace std;

class JsonFile {
public:
    JsonFile();
    void clear();
    string str();

    void add_key(string sKey, string sVal);
    void add_key(string sKey, int iVal);
    void add_key(string sKey, float fVal);
    void add_key(string sKey, bool bVal);

private:
    string find_key(string s, string sKey);
    void split(string s, vector<string>& vsItems, char cDelimiter=' ');

    map<string, string> _allPairs;
};

#endif
