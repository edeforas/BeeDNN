# BeeDNN is a Deep Learning library in C++.

(There is also an prototyping python code to test quickly new ideas).

Compilation projects are using VisualStudio2019 or CMake.

## 0/ Prerequisite:

A C++ compiler.
This tutorial is using VisualStudio2019, any recent version is ok.
This code can also be compiled under GCC or CLang
To compile, use the vs2019 project files or the CMake files.
With vs2019 the CMake files can be used if needed.
The sample codes are commented and are mostly self explanatory.


## 1/ First classification. The sample is sample_classification_xor

For the first classification, double clic on the all.sln solution file in the src/ folder, then:

Compile the sample `sample_classification_xor`, run.
You should see the text :
 
0_xor_0=0
0_xor_1=1
1_xor_0=1
1_xor_1=0
Test succeded.

( This sample learn to reproduce a XOR gate )

Congratulation! You built, trained and computed the output, of your first Neural Network. 
You can try also the sample: `sample_regression_sin`, for regression tasks


## 2/ A more advanced classification (MNIST digits dataset)

The previous XOR sample is using the simple internal Matrix Library, but for heavy learning tasks, use the eigen library.
Eigen is a header only library, with no dependencies.
BeeDNN can use either the internal matrix library or Eigen, to change, set an environnement variable.

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
Best solutions are saved on disk, in  .txt files that can be reloaded.

## 5/ Times Series
Todo

## 6/ Contributing
Any help welcome!
For contributing idea and commits, please see the project github issues, or by any other means.

Todo