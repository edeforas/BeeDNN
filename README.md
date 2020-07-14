# BeeDNN

BeeDNN is a deep learning library.

The git repository is https://github.com/edeforas/BeeDNN

The API is clear and simple, the goal is that every C++ coder can contribute, test, and use Deep Neural Network.
BeeDNN can run on small devices. It is even possible to learn directly on the device.

Please see at: https://github.com/edeforas/BeeDNN/issues for contributing ideas.
No dependencies needed, every algorithm rewritten in C++ from scratch.
To increase speed, ones can choose the Eigen library (http://eigen.tuxfamily.org), instead of the internal matrix library.
There is also a GUI application (in Qt) for live tests.

Layers:
- Dense, with or without bias
- GlobalGain, GlobalBias GlobalAffine, Gain, Bias, Affine
- Convolution2D, PoolMax2D, ChannelBias
- Softmax, Softmin
- PRelu, RRelu
- Layers and activations are decoupled and can be in any order

Activations (in alphabetical order):
- Absolute, Asinh, Atan
- Bent, BinaryStep, Bipolar, BipolarSigmoid
- ComplementaryLogLog
- dSiLU
- ELiSH, Elliot, ELU, Exponential, E2RU, E3RU
- FTS, FTS+
- Gauss, GELU
- HardELU, HardSigmoid, HardShrink, HardTanh
- ISRELU
- Linear, LeakyRelu, LeakyRelu256, LecunTanh, LiSHT, Logit, LogSigmoid
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

Overfitting:
- Layers: Dropout, GaussianNoise, GaussianDropout, UniformNoise
- Regularizer: GradientClip, GradientClipTanh, L1, L2

Data augmentation
- WIP

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
