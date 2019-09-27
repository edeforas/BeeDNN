# BeeDNN

BeeDNN is a deep learning library.

The goal is to have a minimal, clear, and simple API and samples so everybody can contribute, test, and use DNN.
No dependencies needed, every algorithm rewritten in C++ from scratch. There is also a GUI application for live tests.

Implemented so far:
- Dense layer, with or without bias
- Dropout layer, GaussianNoise layer, GaussianDropout layer
- GlobalGain layer , GlobalBias layer
- PoolAveraging1D layer (WIP)
- Softmax layer
- PRelu layer
- Class balancing
- layers and activations are decoupled and can be in any orders
- mini batch learn, SGD learn, batch learn
- optimizers: SGD, Momentum, Nesterov, Adam, Nadam, Adagrad, Adamax, RMSprop, RPROP-, iRPROP-
- classification or regression, test and/or learn
- many activation functions: Absolute, Asinh, Atan, Bent, BinaryStep, Bipolar, BipolarSigmoid, ComplementaryLogLog, Elliot, ELU, Exponential, Gauss, HardELU, HardSigmoid, HardShrink, HardTanh, ISRELU, Linear, LeakyRelu, LeakyRelu256, LecunTanh, Logit, LogSigmoid, TwiceLeakyRelu6, NLRelu, Parablu, Relu, Relu6, Selu, SQNL, SoftPlus, Sin, SinC, Sigmoid, Swish, SoftShrink, SoftSign, Tanh, ThresholdedRelu
- Loss functions can be: MeanSquareError, MeanAbsoluteError, L2, L1, LogCosh, CrossEntropy or BinaryCrossEntropy
- optional eigen use (http://eigen.tuxfamily.org), or use internal matrix library
- network and weights are saved in a plain .txt file
- all in simple C++

The companion GUI app use Qt, compiled binaries coming soon
