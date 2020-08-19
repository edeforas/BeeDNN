#ifndef DataSourceFunctions_
#define DataSourceFunctions_

#include <string>
using namespace std;

#include "Matrix.h"
#include "DataSourceSelector.h"

class DataSourceFunctions : public DataSourceSelector
{
public:
	DataSourceFunctions();
    ~DataSourceFunctions();

    virtual bool load(const string & sName) override;

private:
    void load_function();
    void load_and();
    void load_xor();

    float get_function_val(float x) const;
};

#endif
