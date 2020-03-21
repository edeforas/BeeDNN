# BeeDNN

BeeDNN is a deep learning library.

The API is clear and simple, the goal is that everybody can contribute, test, and use DNN in C++.
BeeDNN can run on small devices. It is even possible to learn or do knowledge transfer directly on the device.
Please see at: https://github.com/edeforas/BeeDNN/issues for contributing ideas.
No dependencies needed, every algorithm rewritten in C++ from scratch.
To increase speed, ones can choose the Eigen library (http://eigen.tuxfamily.org), instead of the internal matrix library.
There is also a GUI application (in Qt) for live tests.

Layers:
- Dense, with or without bias
- Dropout, GaussianNoise, GaussianDropout, UniformNoise
- Gain, GlobalGain, Bias, GlobalBias
- Convolution2D, PoolMax2D, ChannelBias
- Softmax
- PRelu, RRelu
- Layers and activations are decoupled and can be in any order

Activations (in alphabetical order):
- Absolute, Asinh, Atan
- Bent, BinaryStep, Bipolar, BipolarSigmoid
- ComplementaryLogLog
- dSiLU
- ELiSH, Elliot, ELU, Exponential, E2RU, E3RU
- Gauss, GELU
- HardELU, HardSigmoid, HardShrink, HardTanh
- ISRELU
- Linear, LeakyRelu, LeakyRelu256, LecunTanh, Logit, LogSigmoid
- TwiceLeakyRelu6
- Mish
- NLRelu
- O2RU, O3RU
- Parablu
- Relu, Relu6
- Selu, SQNL, SQ-RBF, SoftPlus, Sin, SinC, Sigmoid, SiLU, Swish, SoftShrink, SoftSign
- Tanh, TanhShrink, ThresholdedRelu

Loss functions: 
- MeanSquareError, MeanAbsoluteError
- L2, L1
- LogCosh
- SparseCategoricalCrossEntropy, CategoricalCrossEntropy, BinaryCrossEntropy

Regularizer:
- Clamp
- Tanh

Learning and optimization:
- Classification or regression, test and/or learn
- SGD, Momentum, Nesterov
- Adam, Nadam
- Adagrad
- Adamax
- RMSprop
- RPROP-, iRPROP-
- MetaOptimizer (V1)
- Class balancing if needed
- Keep best model vs epochs
- LearnMore mode 
- Reboost mode

I/O:
- MNIST reader
- CIFAR10 reader
- csv file reader
- Network, weights and training parameters are saved in a simple .txt file
