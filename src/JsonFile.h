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

class JsonFileWriter {
public:
    JsonFileWriter();

    void clear();
    string to_string();
    void save(const string& sFile);

    void enter_section(const string& sSection);
    void leave_section();

    void add(const string& sKey, const string& sVal);
    void add(const string& sKey, int iVal);
    void add(const string& sKey, float fVal);
    void add(const string& sKey, bool bVal);

    void add_array(const string& sKey, int iSize, const float* pVal);

private:
    void add_string(const string& sKey, const string& s);
    
    bool _bPendingComma;
    string _sSectionIndent;
    string _sOut;
};

#endif
