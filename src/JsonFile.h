/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#ifndef JsonFile_
#define JsonFile_

#include <string>
using namespace std;

class JsonFile {
public:
    JsonFile();

    void clear();
    string to_string();
    void save(string sFile);

    void enter_section(string sSection);
    void leave_section();

    void add(string sKey, string sVal);
    void add(string sKey, int iVal);
    void add(string sKey, float fVal);
    void add(string sKey, bool bVal);

    void add_array(string sKey, int iSize, const float* pVal);
private:
    void add_string(const string& sKey, string s);
    
    bool _bPendingComma;
    string _sSectionIndent;
    string _sOut;
};

#endif
