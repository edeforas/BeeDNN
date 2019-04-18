# BeeDNN

BeeDNN is a library to play/learn with DNN algorithms.

The goal is to have a minimal, clear and concise APi and samples so everybody can contribute and play with DNN.
No dependencies needed, every algorithm rewritten. There is also a simple GUI application for snappy tests.

Implemented so far:
- dense layer, with or without bias
- dropout layer
- global gain layer
- pool averaging 1D layer
- softmax layer
- layers can be in any orders
- mini batch learn, sgd learn
- SGD, Momentum, Nesterov, Adam , Nadam, Adagrad, Adamax, rmsprop
- train or fit
- lot of activation functions
- MeanSquareError, CrossEntropy or BinaryCrossEntropy loss
- tiny-dnn abstraction (optional), so you can compare both library
- eigen (optional) use or internal matrix library
- all in C++

The GUI app use QT, will have binaries soon
