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
	void add_prelu_layer(int inSize);
	void add_softmax_layer();
    void add_dropout_layer(int iSize, float fRatio);
    void add_gaussian_dropout_layer(int iSize, float fProba);
	
	void add_uniform_noise_layer(int iSize, float fNoise);
	void add_gaussian_noise_layer(int iSize, float fStd);
    
	void add_globalgain_layer(int iSize);
    
	void add_globalbias_layer(int iSize);
	void add_bias_layer(int iSize);
	
	void add_poolaveraging1D_layer(int inSize, int outSize);
	void add_poolmax1D_layer(int inSize, int outSize);

    void set_input_size(int iInputSize);
	int output_size() const;
    int input_size() const;

    const vector<Layer*> layers() const;
    Layer& layer(size_t iLayer);
	const Layer& layer(size_t iLayer) const;
	size_t size() const;

	void set_classification_mode(bool bClassificationMode); //true by default
	bool is_classification_mode() const;

	void forward(const MatrixFloat& mIn, MatrixFloat& mOut) const;
	void classify(const MatrixFloat& mIn, MatrixFloat& mClass) const;

    void set_train_mode(bool bTrainMode); // set to true if training, set to false if testing (default)

	bool is_valid(int iInSize, int iOutSize) const; //return true if all size ok

private:
    void update_out_layer_input_size(int& iInSize);
	bool _bTrainMode;
	vector<Layer*> _layers;
    int _iInputSize, _iOutputSize;
	bool _bClassificationMode;
};

#endif
