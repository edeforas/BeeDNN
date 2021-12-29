# BeeDNN Howto

BeeDNN is a Deep Learning library in C++. Train model, predict results, on any platform, with a FPU.

## 0/ Prerequisite:

BeeDNN only need a C++ compiler, no other dependencies.
Projects files are using CMake or VisualStudio2019.
To compile, use the vs2019 project files or the CMake files. This tutorial use the vs2019 files.
The sample codes are commented and are mostly self explanatory.

To speed up training, it is possible to use the eigen library, see in below samples.


## 1/ First classification: XOR classification

This sample learn to reproduce a XOR gate
Launch the all.sln solution file in the src/ folder, then:

Compile the sample `sample_classification_xor`, run.
You should see the text :
 
0_xor_0=0
0_xor_1=1
1_xor_0=1
1_xor_1=0
Test succeded.

Congratulation! You built, trained, and computed the output of your first Neural Network. 
You can try also the sample: `sample_regression_sin`, for regression tasks


## 2/ A more advanced classification: (MNIST digits dataset)+ Dense layers

The previous XOR sample is using the simple internal Matrix Library, but for heavy learning tasks, use the eigen library.
Eigen is a header only library, with no dependencies.
BeeDNN can use either the internal matrix library or Eigen, to select it, set an environnement variable.

To use eigen:

1. Download eigen at: [GNU Eigen](http://eigen.tuxfamily.org/)
2. Unzip
3. Set the environmement variable `EIGEN_PATH` pointing to the eigen/ folder in the unzipped folder, for example in  "C:\dev\eigen-3.4.0"
4. Reload Visual Studio 2019
5. Compile the sample `sample_classification_MNIST` (compile in Release, 64 bit mode, for good performances)

Before running this sample, download the MNIST dataset,
Go to: [MNIST dataset](http://yann.lecun.com/exdb/mnist/) , download and unzip the 4 .gz files (for example with [7-Zip](https://www.7-zip.org/), put the uzipped files (*.ubytes) near the sample executable.

Now, you can run `sample_classification_MNIST`, accuracy is >98.1% after 15 epochs, 2s by epochs.
The learning speed and performances are on pair with TensorFlow using Dense layers.

## 3/ MNIST with all convolutional layers

Use the sample `sample_classification_MNIST_all_convolutional`, build and run
This sample is similar to  the previous sample, but uses convolutional layers, the accuracy goes up to 99% and uses is 20s by epochs.

## 4/ Meta optimisation

The sample is `sample_MetaOptimizer_MNIST`.
It launch several train in parallel (uses all cpu core), and keep the best solutions.
Best solutions are saved on disk, in  .json files that can be reloaded.

## 5/ Times Series
Todo

## 6/ Contributing
Any help welcome!
For contributing idea and commits, please see the project github issues, or by any other means.

Todo