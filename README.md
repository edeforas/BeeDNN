# BeeDNN

BeeDNN is a deep learning library.

The goal is to have a minimal, clear, and simple API and samples so everybody can contribute and play with DNN.
No dependencies needed, every algorithm rewritten from scratch. There is also a GUI application for live tests.

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
- lot of activation functions
- MeanSquareError, CrossEntropy or BinaryCrossEntropy loss
- tiny-dnn abstraction (optional), so you can compare both library
- eigen (optional) use or internal matrix library
- all in C++

The GUI app use QT, will have binaries soon
