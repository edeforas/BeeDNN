# BeeDNN

BeeDNN is a deep learning library.

The goal is to have a minimal, clear, and simple API and samples so everybody can contribute, test, and use DNN.
No dependencies needed, every algorithm rewritten in C++ from scratch. There is also a GUI application for live tests.

Implemented so far:
- dense layer, with or without bias
- dropout layer, gaussian noise layer
- global gain layer
- pool averaging 1D layer
- softmax layer (WIP)
- layers and activations are decoupled, can be in any orders
- mini batch learn, SGD learn, batch learn
- SGD, Momentum, Nesterov, Adam , Nadam, Adagrad, Adamax, RMSprop
- classification or regression
- lot of activation functions: Asinh, Atan, Bent, Elliot, Elu, Exponential, HardSigmoid, Gauss, Linear, LeakyRelu, LeakyRelu256, NLRelu, Parablu, Relu, Selu, SQNL, SoftPlus, Sin, SinC, Sigmoid, Swish, SoftSign, Tanh
- Loss functions can be: MeanSquareError, MeanAbsoluteError, L2, L1, CrossEntropy or BinaryCrossEntropy
- optional eigen use (http://eigen.tuxfamily.org), or use internal matrix library
- all in C++

The GUI app use Qt, compiled binaries coming soon
