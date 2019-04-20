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
- layers can be in any orders
- mini batch learn, sgd learn, batch learn
- SGD, Momentum, Nesterov, Adam , Nadam, Adagrad, Adamax, RMSprop
- classification or regression
- lot of activation functions: Asinh, Atan, Bent, Elliot, Elu, HardSigmoid, Gauss, Linear, LeakyRelu, Parablu, Relu, Selu, SQNL, SoftPlus, Sin, SinC, Sigmoid, Swish, SoftSign, Tanh
- MeanSquareError, CrossEntropy or BinaryCrossEntropy loss
- optional tiny-dnn (https://github.com/tiny-dnn) bindining, so you can compare both library
- eigen (http://eigen.tuxfamily.org) (optional) use or internal matrix library
- all in C++

The GUI app use Qt, binaries coming soon
