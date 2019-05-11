# BeeDNN

BeeDNN is a deep learning library.

The goal is to have a minimal, clear, and simple API and samples so everybody can contribute, test, and use DNN.
No dependencies needed, every algorithm rewritten in C++ from scratch. There is also a GUI application for live tests.

Implemented so far:
- dense layer, with or without bias
- dropout layer
- global gain layer
- pool averaging 1D layer
- softmax layer
- layers and activations are decoupled, can be in any orders
- mini batch learn, SGD learn, batch learn
- SGD, Momentum, Nesterov, Adam , Nadam, Adagrad, Adamax, RMSprop
- classification or regression
- lot of activation functions: Asinh, Atan, Bent, Elliot, Elu, HardSigmoid, Gauss, Linear, LeakyRelu, NLRelu, Parablu, Relu, Selu, SQNL, SoftPlus, Sin, SinC, Sigmoid, Swish, SoftSign, Tanh
- Loss functions can be: MeanSquareError, MeanAbsoluteError, L2, L1, CrossEntropy or BinaryCrossEntropy
- optional tiny-dnn binding (https://github.com/tiny-dnn), so you can compare both library
- optional eigen use (http://eigen.tuxfamily.org), or use internal matrix library
- all in C++

The GUI app use Qt, binaries coming soon
