#include "DataSourceFunctions.h"

////////////////////////////////////////////////////////////////////////
DataSourceFunctions::DataSourceFunctions() : DataSourceSelector()
{ }
////////////////////////////////////////////////////////////////////////
DataSourceFunctions::~DataSourceFunctions()
{}
////////////////////////////////////////////////////////////////////////
bool DataSourceFunctions::load(const string& sName)
{
	if (DataSourceSelector::load(sName))
		return true;

	_sName = sName;
	if (sName == "And")
		load_and();
	else if (sName == "Xor")
		load_xor();
	else
		load_function();
	
	return true;
}
////////////////////////////////////////////////////////////////////////
void DataSourceFunctions::load_and()
{
    _mTrainData.resize(4,2);
    _mTrainData(0,0)=0; _mTrainData(0,1)=0;
    _mTrainData(1,0)=1; _mTrainData(1,1)=0;
    _mTrainData(2,0)=0; _mTrainData(2,1)=1;
    _mTrainData(3,0)=1; _mTrainData(3,1)=1;

    _mTrainTruth.resize(4,1);
    _mTrainTruth(0,0)=0;
    _mTrainTruth(1,0)=0;
    _mTrainTruth(2,0)=0;
    _mTrainTruth(3,0)=1;

    _mTestData=_mTrainData;
    _mTestTruth=_mTrainTruth;

    _bHasTrainData=true;
    _bHasTestData=false;
}
////////////////////////////////////////////////////////////////////////
void DataSourceFunctions::load_xor()
{
    _mTrainData.resize(4,2);
    _mTrainData(0,0)=0; _mTrainData(0,1)=0;
    _mTrainData(1,0)=1; _mTrainData(1,1)=0;
    _mTrainData(2,0)=0; _mTrainData(2,1)=1;
    _mTrainData(3,0)=1; _mTrainData(3,1)=1;

    _mTrainTruth.resize(4,1);
    _mTrainTruth(0,0)=0;
    _mTrainTruth(1,0)=1;
    _mTrainTruth(2,0)=1;
    _mTrainTruth(3,0)=0;

    _mTestData=_mTrainData;
    _mTestTruth=_mTrainTruth;

    _bHasTrainData=true;
    _bHasTestData=false;
}
////////////////////////////////////////////////////////////////////////
void DataSourceFunctions::load_function()
{
    float fMin=-4.f;
    float fMax=4.f;

    int iNbPointsLearn = 100;
    _mTrainData.resize(iNbPointsLearn,1);
    _mTrainTruth.resize(iNbPointsLearn,1);
    float dStep=(fMax-fMin)/(iNbPointsLearn-1.f);
    float fVal=fMin;
    for( int i=0;i<iNbPointsLearn;i++)
    {
        float fOut = get_function_val(fVal);
        _mTrainData(i)=fVal;
        _mTrainTruth(i)=fOut;
        fVal+=dStep;
    }

    int iNbPointsTest = 199;
    _mTestData.resize(iNbPointsTest, 1);
    _mTestTruth.resize(iNbPointsTest, 1);
    dStep = (fMax - fMin) / (iNbPointsTest - 1.f);
    fVal = fMin;
    for (int i = 0; i < iNbPointsTest; i++)
    {
        float fOut = get_function_val(fVal);
        _mTestData(i) = fVal;
        _mTestTruth(i) = fOut;
        fVal += dStep;
    }

    _bHasTrainData=true;
    _bHasTestData= true;
}
////////////////////////////////////////////////////////////////////////
float DataSourceFunctions::get_function_val(float x) const
{
    if (_sName == "Identity")
        return x;

    if (_sName == "Sin")
        return sinf(x);

    if (_sName == "Sin4Period")
        return sinf(x*4.f);

    if (_sName == "Abs")
        return fabs(x);

    if (_sName == "Parabolic")
        return x * x;

    if (_sName == "Exp")
        return expf(x);

    if (_sName == "Gauss")
        return expf(-x * x);

    if (_sName == "Rectangular")
        return (float)(((((int)x) + (x < 0.f)) + 1) & 1);

    return 0.f;
}
////////////////////////////////////////////////////////////////////////
