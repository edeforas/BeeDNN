#include "DataSource.h"

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
void DataSource::load_mnist()
{
    if(_sLastLoaded=="MNIST")
        return;

    _sLastLoaded="MNIST";

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
     _sLastLoaded="and";

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
    _sLastLoaded="xor";

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
    if(_sLastLoaded=="TextFile")
        return;

    _sLastLoaded="TextFile";

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
void DataSource::load_function(string sFunction,float fMin, float fMax, int iNbPoints)
{
    _sLastLoaded=sFunction;

    _mTrainData.resize(iNbPoints,1);
    _mTrainAnnotation.resize(iNbPoints,1);

    float dStep=(fMax-fMin)/(iNbPoints-1.f);

    float fVal=fMin,fOut=0.f;

    for( int i=0;i<iNbPoints;i++)
    {
        if(sFunction=="Identity")
            fOut=fVal;

        if(sFunction=="Sin")
            fOut=sinf(fVal);

        if(sFunction=="Abs")
            fOut=fabs(fVal);

        if(sFunction=="Parabolic")
            fOut=fVal*fVal;

        if(sFunction=="Gamma")
            fOut=tgammaf(fVal);

        if(sFunction=="Exp")
            fOut=expf(fVal);

        if(sFunction=="Sqrt")
            fOut=sqrtf(fVal);

        if(sFunction=="Ln")
            fOut=logf(fVal);

        if(sFunction=="Gauss")
            fOut=expf(-fVal*fVal);

        if(sFunction=="Inverse")
            fOut=1.f/fVal;

        if(sFunction=="Rectangular")
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
