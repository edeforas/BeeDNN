#include "NetTrain.h"

#include "Net.h"
#include "Layer.h"
#include "Matrix.h"

/////////////////////////////////////////////////////////////////////////////////////////////////
NetTrain::NetTrain()
{ }
/////////////////////////////////////////////////////////////////////////////////////////////////
NetTrain::~NetTrain()
{ }
/////////////////////////////////////////////////////////////////////////////////////////////////
double NetTrain::compute_loss(const Net& net, const MatrixFloat &mSamples, const MatrixFloat &mTruth)
{
	if(net.layers().size()==0)
        return -1.;

    int iNbSamples=(int)mSamples.rows();
    MatrixFloat mOut,mError;

    for(int i=0;i<iNbSamples;i++)
    {
        net.forward(mSamples.row(i),mOut);

        if(i==0)
            mError=(mOut-mTruth.row(i)).cwiseAbs2();
        else
            mError+=(mOut-mTruth.row(i)).cwiseAbs2();
    }

    return mError.sum()/(double)iNbSamples;
}
/////////////////////////////////////////////////////////////////////////////////////////////
vector<double> NetTrain::loss()
{
    return _vdLoss;
}
/////////////////////////////////////////////////////////////////////////////////////////////////
void NetTrain::train(Net& net,const MatrixFloat& mSamples,const MatrixFloat& mTruthLabel,const TrainOption& topt)
{
    //create a kronecker matrix, one line by sample
    int iMax=(int)mTruthLabel.maxCoeff();
    MatrixFloat mTruth(mTruthLabel.rows(),iMax+1);
    mTruth.setZero();

    for(int i=0;i<mTruth.rows();i++)
        mTruth(i,(int)mTruthLabel(i,0))=1;

    fit(net,mSamples,mTruth,topt);
}
/////////////////////////////////////////////////////////////////////////////////////////////////