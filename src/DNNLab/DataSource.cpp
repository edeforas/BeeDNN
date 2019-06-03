#include "DataSource.h"

#include "NetUtil.h"
#include "MNISTReader.h"

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

    if(sName=="MNIST")
        load_mnist();

    else if(sName=="and")
        load_and();

    else if(sName=="xor")
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
    r.read_from_folder(".",_mTrainData,_mTrainAnnotation,_mTestData,_mTestAnnotation);
    _mTrainData/=255.f;
    _mTestData/=255.f;

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

    _mTrainAnnotation.resize(4,1);
    _mTrainAnnotation(0,0)=0;
    _mTrainAnnotation(1,0)=0;
    _mTrainAnnotation(2,0)=0;
    _mTrainAnnotation(3,0)=1;

    _mTestData=_mTrainData;
    _mTestAnnotation=_mTrainAnnotation;

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

    _mTrainAnnotation.resize(4,1);
    _mTrainAnnotation(0,0)=0;
    _mTrainAnnotation(1,0)=1;
    _mTrainAnnotation(2,0)=1;
    _mTrainAnnotation(3,0)=0;

    _mTestData=_mTrainData;
    _mTestAnnotation=_mTrainAnnotation;

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
    _mTrainAnnotation=fromFile("train_truth.txt");

    _mTestData=fromFile("test_data.txt");
    _mTestAnnotation=fromFile("test_truth.txt");

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
    _mTrainAnnotation=fromFile("Fisher_truth.txt");

    _mTestData=_mTrainData;
    _mTestAnnotation=_mTrainAnnotation;

    _bHasTrainData=true;
    _bHasTestData=false;
}
////////////////////////////////////////////////////////////////////////
void DataSource::load_function()
{
    float fMin=-4.f;
    float fMax=4.f;
    int iNbPoints=100;

    _mTrainData.resize(iNbPoints,1);
    _mTrainAnnotation.resize(iNbPoints,1);

    float dStep=(fMax-fMin)/(iNbPoints-1.f);

    float fVal=fMin,fOut=0.f;

    for( int i=0;i<iNbPoints;i++)
    {
        if(_sName=="Identity")
            fOut=fVal;

        if(_sName=="Sin")
            fOut=sinf(fVal);

        if(_sName=="Abs")
            fOut=fabs(fVal);

        if(_sName=="Parabolic")
            fOut=fVal*fVal;

        if(_sName=="Gamma")
            fOut=tgammaf(fVal);

        if(_sName=="Exp")
            fOut=expf(fVal);

        if(_sName=="Sqrt")
            fOut=sqrtf(fVal);

        if(_sName=="Ln")
            fOut=logf(fVal);

        if(_sName=="Gauss")
            fOut=expf(-fVal*fVal);

        if(_sName=="Inverse")
            fOut=1.f/fVal;

        if(_sName=="Rectangular")
            fOut= (float)(((((int)fVal)+(fVal<0.f))+1) & 1 );

        _mTrainData(i)=fVal;
        _mTrainAnnotation(i)=fOut;
        fVal+=dStep;
    }

    _mTestData=_mTrainData;
    _mTestAnnotation=_mTrainAnnotation;

    _bHasTrainData=true;
    _bHasTestData=false;
}
////////////////////////////////////////////////////////////////////////
const MatrixFloat& DataSource::train_data() const
{
    return _mTrainData;
}
////////////////////////////////////////////////////////////////////////
const MatrixFloat& DataSource::train_annotation() const
{
    return _mTrainAnnotation;
}
////////////////////////////////////////////////////////////////////////
const MatrixFloat& DataSource::test_data() const
{
    return _mTestData;
}
////////////////////////////////////////////////////////////////////////
const MatrixFloat& DataSource::test_annotation() const
{
    return _mTestAnnotation;
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
int DataSource::data_cols() const
{
    return (int)_mTrainData.cols();
}
////////////////////////////////////////////////////////////////////////
int DataSource::annotation_cols() const
{
    return (int)_mTrainAnnotation.cols();
}
////////////////////////////////////////////////////////////////////////
void DataSource::clear()
{
    _mTrainData.resize(0,0);
    _mTrainAnnotation.resize(0,0);
    _mTestData.resize(0,0);
    _mTestAnnotation.resize(0,0);

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
