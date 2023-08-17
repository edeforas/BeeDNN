# BeeDNN

BeeDNN is a deep learning library.

The git repository is https://github.com/edeforas/BeeDNN

The API is clear and simple, the goal is that every C++ developer can contribute, test, and use Deep Neural Network.
BeeDNN can run on small devices. It is even possible to learn directly on the device.

Please see at: https://github.com/edeforas/BeeDNN/issues for contributing ideas.
No dependencies needed, every algorithm rewritten in C++ from scratch.
To increase speed, ones can choose the Eigen library (http://eigen.tuxfamily.org), instead of the internal matrix library.

Initializers:
- GlorotUniform, GlorotNormal
- HeUniform, HeNormal
- LecunUniform, LecunNormal
- Zeros, Ones

Layers:
- Dense (with bias), Dot (without bias)
- GlobalGain, GlobalBias, GlobalAffine, Gain, Bias, Affine
- Softmax, Softmin
- PRelu, RRelu, PELU, TERELU, CRelu
- Gated activations: GLU, ReGLU, Bilinear, SwiGLU, GEGLU, GTU, SeGLU
- Layers and activations are decoupled and can be in any order

Time series:
- TimeDistributedBias
- TimeDistributedDot
- TimeDistributedDense
- WIP

2D layers:
- Convolution2D, ChannelBias
- MaxPool2D, GlobalMaxPool2D, 
- AveragePooling2D, GlobalAveragePooling2D
- ZeroPadding2D

Activations (in alphabetical order):
- Absolute, Asinh, Atan
- Bent, BinaryStep, Bipolar, BipolarSigmoid, Bump
- ComplementaryLogLog, CELU
- dSiLU
- ELiSH, Elliot, ELU, Exponential, E2RU, E3RU, Eswish
- FTS, FTS+
- Gauss, GELU
- HardELU, HardSigmoid, HardShrink, HardTanh, HardSwish, Hann
- ISRELU
- Linear, LeakyRelu, LeakyRelu256, LecunTanh, LiSHT, Logit, LogSigmoid
- Mish
- NLRelu
- O2RU, O3RU
- SmoothSoftPlus
- SmoothTanh
- SmoothSigmoid
- Relu, Relu6
- Selu, SQNL, SQ-RBF, SoftPlus, Sin, SinC, Sigmoid, SiLU, Swish, SoftShrink, SoftSign, SoftSteps, SineReLU
- Tanh, TanhExp, TanhShrink, ThresholdedRelu,TwiceLeakyRelu6

Loss functions: 
- MeanSquareError, MeanAbsoluteError, MeanCubicError
- L2, L1, L3
- LogCosh
- Huber, PseudoHuber
- SparseCategoricalCrossEntropy, CategoricalCrossEntropy, BinaryCrossEntropy

Overfitting:
- Layers: Dropout, GaussianNoise, GaussianDropout, UniformNoise
- Regularizer: GradientClip, GradientNormClip, GradientClipTanh, L1, L2, L1L2

Data augmentation
- RandomFlip

Learning and optimization:
- Classification or regression, test and/or learn
- SGD, Momentum, MomentumNg, Nesterov
- Adam, AdamW, Nadam
- Adagrad
- Adamax
- Amsgrad
- RMSprop
- RPROP-, iRPROP-
- MetaOptimizer (V1)
- Class balancing if needed
- Keep best model vs epochs
- LearnMore mode 
- Reboost mode

KMeans:
- can use any loss
- batchmode learning

I/O:
- MNIST reader
- CIFAR10 reader
- csv file reader
- model, weights and training parameters are saved in a simple .json file

Precomputing:
- StandardScaler, MinMaxScaler
	
Commented samples:
- Simple XOR classification, with and w/o softmax
- Simple sinus regression
- MNIST with dense net
- MNIST using time serie (a time frame is an image row)
- MNIST with poolmax2D
- MNIST all convolutional
- MNIST and Meta Optimizer: select best activation
- CIFAR10 conv2D using poolmax2D
- MNIST with kmeans and custom loss

Build with vs2019 or CMake.
To compile, run the samples, etc, please read the HOWTO.md file