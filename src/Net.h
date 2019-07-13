/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#ifndef Net_
#define Net_

#include <vector>
#include <string>
using namespace std;

class Layer;
#include "Matrix.h"

class Net
{
public:
    Net();
    virtual ~Net();
    Net& operator=(const Net& other);

	void clear();
	void init();

    void add_dense_layer(int inSize, int outSize, bool bHasBias=true);
	void add_activation_layer(string sType);
	void add_softmax_layer();
    void add_dropout_layer(int iSize, float fRatio);
    void add_gaussian_noise_layer(int iSize, float fStd);
    void add_globalgain_layer(int inSize, float fGlobalGain);
    void add_poolaveraging1D_layer(int inSize, int outSize);

    void set_input_size(int iInputSize);
	int output_size() const;
    int input_size() const;

    const vector<Layer*> layers() const;
    Layer& layer(size_t iLayer);
	const Layer& layer(size_t iLayer) const;
	size_t size() const;

    void forward(const MatrixFloat& mIn,MatrixFloat& mOut) const;
    int classify(const MatrixFloat& mIn) const;
    void classify_all(const MatrixFloat& mIn, MatrixFloat& mClass) const;

    void set_train_mode(bool bTrainMode); // set to true if training, set to false if testing

private:
    void update_out_layer_input_size(int& iInSize);
	bool _bTrainMode;
	vector<Layer*> _layers;
    int _iInputSize, _iOutputSize;
};

#endif
