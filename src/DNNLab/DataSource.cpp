#include "DataSource.h"

#include "NetUtil.h"
#include "MNISTReader.h"

////////////////////////////////////////////////////////////////////////
void replace_last(string& s, string sOld,string sNew)
{
    auto found = s.rfind(sOld);
    if(found != std::string::npos)
        s.replace(found, sOld.length(), sNew);
}
////////////////////////////////////////////////////////////////////////
DataSource::DataSource()
{
    _bHasTrainData=false;
    _bHasTestData=false;
}
////////////////////////////////////////////////////////////////////////
DataSource::~DataSource()
{}
////////////////////////////////////////////////////////////////////////
void DataSource::write(string& s)
{
    s+=string("DataSource=")+_sName+string("\n");
}
////////////////////////////////////////////////////////////////////////
void DataSource::read(const string& s)
{
    string sVal=NetUtil::find_key(s,"DataSource");

    load(sVal);
}
////////////////////////////////////////////////////////////////////////
void DataSource::load(const string& sName)
{
    clear();

    if(sName.empty())
        return;

    // if has an extention -> custom file
    if(sName.find('.')!=string::npos)
    {
        //create 4 file names (can no exist)
        string sTrainData=sName;
        string sTrainTruth=sName;
        string sTestData=sName;
        string sTestTruth=sName;

        //create sTrainData
        replace_last(sTrainData,"test","train");
        replace_last(sTrainData,"truth","data");

        //create sTrainTruth
        replace_last(sTrainTruth,"test","train");
        replace_last(sTrainTruth,"data","truth");

        //create sTestData
        replace_last(sTestData,"train","test");
        replace_last(sTestData,"truth","data");

        //create sTestTruth
        replace_last(sTestTruth,"train","test");
        replace_last(sTestTruth,"data","truth");

        //try to load every file
        _mTrainData=fromFile(sTrainData);
        _mTrainTruth=fromFile(sTrainTruth);
        _mTestData=fromFile(sTestData);
        _mTestTruth=fromFile(sTestTruth);

        _bHasTrainData=(_mTrainData.size()!=0) && (_mTrainTruth.size()!=0) ;
        _bHasTestData=(_mTestData.size()!=0) && (_mTestTruth.size()!=0) && (sTrainData!=sTestData) && (sTrainTruth!=sTestTruth);

        if(!_bHasTestData)
        {
            _mTestData.resize(0,0);
            _mTestTruth.resize(0,0);
        }

        _sName=sName;

        return;
    }

    if(sName=="MNIST")
        load_mnist();

    else if(sName=="And")
        load_and();

    else if(sName=="Xor")
        load_xor();

    else if(sName=="TextFile")
        load_textfile();

    else if(sName=="Fisher")
        load_fisher();

    else
    {
        _sName=sName;
        load_function();
    }
    _sName=sName;
}
////////////////////////////////////////////////////////////////////////
void DataSource::load_mnist()
{
    if(_sName=="MNIST")
        return;

    _sName="MNIST";

    MNISTReader r;
    r.read_from_folder(".",_mTrainData,_mTrainTruth,_mTestData,_mTestTruth);
    _mTrainData/=256.f;
    _mTestData/=256.f;

    _bHasTrainData=true;
    _bHasTestData=true;
}
////////////////////////////////////////////////////////////////////////
void DataSource::load_and()
{
    _sName="and";

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
void DataSource::load_xor()
{
    _sName="xor";

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
void DataSource::load_textfile()
{
    if(_sName=="TextFile")
        return;

    _sName="TextFile";

    _mTrainData=fromFile("train_data.txt");
    _mTrainTruth=fromFile("train_truth.txt");

    _mTestData=fromFile("test_data.txt");
    _mTestTruth=fromFile("test_truth.txt");

    _mTrainData/=255.f; //for now
    _mTestData/=255.f; //for now;

    _bHasTrainData=true;
    _bHasTestData=true;
}
////////////////////////////////////////////////////////////////////////
void DataSource::load_fisher()
{
    if(_sName=="Fisher")
        return;

    _sName="Fisher";

    _mTrainData=fromFile("Fisher_data.txt");
    _mTrainTruth=fromFile("Fisher_truth.txt");

    _mTestData=_mTrainData;
    _mTestTruth=_mTrainTruth;

    _bHasTrainData=true;
    _bHasTestData=false;
}
////////////////////////////////////////////////////////////////////////
void DataSource::load_function()
{
    float fMin=-4.f;
    float fMax=4.f;

    int iNbPointsLearn = 100;
    _mTrainData.resize(iNbPointsLearn,1);
    _mTrainTruth.resize(iNbPointsLearn,1);
    float dStep=(fMax-fMin)/(iNbPointsLearn-1.f);
    float fVal=fMin,fOut=0.f;
    for( int i=0;i<iNbPointsLearn;i++)
    {
		fOut = get_function_val(fVal);
        _mTrainData(i)=fVal;
        _mTrainTruth(i)=fOut;
        fVal+=dStep;
    }

	int iNbPointsTest = 199;
	_mTestData.resize(iNbPointsTest, 1);
	_mTestTruth.resize(iNbPointsTest, 1);
	dStep = (fMax - fMin) / (iNbPointsTest - 1.f);
	fVal = fMin, fOut = 0.f;
	for (int i = 0; i < iNbPointsTest; i++)
	{
		fOut = get_function_val(fVal);
		_mTestData(i) = fVal;
		_mTestTruth(i) = fOut;
		fVal += dStep;
	}

    _bHasTrainData=true;
    _bHasTestData= true;
}
////////////////////////////////////////////////////////////////////////
float DataSource::get_function_val(float x)
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

	if (_sName == "Gamma")
		return tgammaf(x);

	if (_sName == "Exp")
		return expf(x);

	if (_sName == "Gauss")
		return expf(-x * x);

	if (_sName == "Rectangular")
		return (float)(((((int)x) + (x < 0.f)) + 1) & 1);

	return 0.f;
}
////////////////////////////////////////////////////////////////////////
const MatrixFloat& DataSource::train_data() const
{
    return _mTrainData;
}
////////////////////////////////////////////////////////////////////////
const MatrixFloat& DataSource::train_truth() const
{
    return _mTrainTruth;
}
////////////////////////////////////////////////////////////////////////
const MatrixFloat& DataSource::test_data() const
{
    return _mTestData;
}
////////////////////////////////////////////////////////////////////////
const MatrixFloat& DataSource::test_truth() const
{
    return _mTestTruth;
}
////////////////////////////////////////////////////////////////////////
bool DataSource::has_data() const
{
    return _bHasTrainData || _bHasTestData;
}
////////////////////////////////////////////////////////////////////////
bool DataSource::has_train_data() const
{
    return _bHasTrainData;
}
////////////////////////////////////////////////////////////////////////
bool DataSource::has_test_data() const
{
    return _bHasTestData;
}
////////////////////////////////////////////////////////////////////////
int DataSource::data_size() const
{
    return (int)_mTrainData.cols();
}
////////////////////////////////////////////////////////////////////////
int DataSource::annotation_cols() const
{
    return (int)_mTrainTruth.cols();
}
////////////////////////////////////////////////////////////////////////
void DataSource::clear()
{
    _mTrainData.resize(0,0);
    _mTrainTruth.resize(0,0);
    _mTestData.resize(0,0);
    _mTestTruth.resize(0,0);

    _bHasTestData=false;
    _bHasTrainData=false;

    _sName="";
}
////////////////////////////////////////////////////////////////////////
const string DataSource::name() const
{
    return _sName;
}
////////////////////////////////////////////////////////////////////////
