//this sample is a basic toy regression task.
//the problem here is to approximate a sinus function and to evaluate the model error

#include <iostream>
#include <iomanip>
#include <cmath>
using namespace std;

#include "Net.h"
#include "NetTrain.h"

/////////////////////////////////////////////////////////////////////
// for testU only
inline bool is_near(double a, double b, double tolerancis = 1.e-10)
{
	return fabs(a - b) < tolerancis;
}
void test(bool bTest, string sMessage = "")
{
	if (bTest) return;

	cout << "Test failed: " << sMessage << endl;
	exit(-1);
}
/////////////////////////////////////////////////////////////////////

int main()
{
    //build net
    Net net;
    net.add_dense_layer(1,10);
	net.add_activation_layer("Tanh");
	net.add_dense_layer(10, 1);
	net.set_classification_mode(false); //set regression mode

    //set train data
    MatrixFloat mTruth(128,1);
    MatrixFloat mSamples(128,1);
    for(int i=0;i<128;i++)
    {
        float x=i/100.f;
        mTruth(i,0)=sin(x);
        mSamples(i,0)=x;
    }

    //setup and train net
    cout << "Fitting..." << endl;
	NetTrain netfit;
	netfit.set_epochs(1000);
	netfit.set_net(net);
	netfit.set_train_data(mSamples, mTruth);
	netfit.train();

    //print truth and prediction
	MatrixFloat mPredict;
	net.forward(mSamples, mPredict);
	for (int i = 0; i < mSamples.size(); i += 8)
	{
		cout << setprecision(4) << "x=" << mSamples(i, 0) << "\ttruth=" << mTruth(i, 0) << "\tpredict=" << mPredict(i, 0) << endl;
		test(is_near(mTruth(i, 0), mPredict(i, 0), 0.05));
	}
    //compute and print loss
    float fLoss=netfit.compute_loss(mSamples,mTruth);
    cout << "Loss=" << fLoss << endl;

	cout << "end of test." << endl;
    return 0;
}
