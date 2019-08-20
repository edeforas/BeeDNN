# BeeDNN

BeeDNN is a deep learning library.

The goal is to have a minimal, clear, and simple API and samples so everybody can contribute, test, and use DNN.
No dependencies needed, every algorithm rewritten in C++ from scratch. There is also a GUI application for live tests.

Implemented so far:
- dense layer, with or without bias
- dropout layer, gaussian noise layer
- global gain layer , global bias layer
- pool averaging 1D layer (WIP)
- Softmax layer
- PRelu layer
- layers and activations are decoupled, can be in any orders
- mini batch learn, SGD learn, batch learn
- SGD, Momentum, Nesterov, Adam, Nadam, Adagrad, Adamax, RMSprop
- classification or regression, test and/or learn
- many activation functions: Absolute, Asinh, Atan, Bent, BinaryStep, BipolarSigmoid, ComplementaryLogLog, Elliot, ELU, Exponential, Gauss, HardELU, HardSigmoid, HardShrink, HardTanh, ISRELU, Linear, LeakyRelu, LeakyRelu256, LogSigmoid, TwiceLeakyRelu6, NLRelu, Parablu, Relu, Relu6, Selu, SQNL, SoftPlus, Sin, SinC, Sigmoid, Swish, SoftShrink, SoftSign, Tanh
- Loss functions can be: MeanSquareError, MeanAbsoluteError, L2, L1, LogCosh, CrossEntropy or BinaryCrossEntropy
- optional eigen use (http://eigen.tuxfamily.org), or use internal matrix library
- network and weights are saved in a plain .txt file
- all in simple C++

The companion GUI app use Qt, compiled binaries coming soon
