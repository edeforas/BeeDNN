BeeDNN is a Deep Learing library in C++.
(There is also an prototyping python code to test quickly new ideas).

Compilation projects are using vs2019 or CMake.

Prerequisite:

A C++ compiler, this tutorial will use Visual Studio 2019, any recent version will be ok.
This code can also be compiled under GCC or CLang
To compile, ones can use the vs2019 project files, or CMake files.
With vs2019 the CMake files can also be used.

The sample codes are commented and should self explanatory, but do not hesitate to comment, ask or contribute.


1/ First classification. The sample is sample_classification_xor

For the first classification, double clic on the all.sln solution file in the src/ folder, then:

Compile the sample sample_classification_xor, run.
You should see the text :
 
0_xor_0=0
0_xor_1=1
1_xor_0=1
1_xor_1=0
Test succeded.

( This sample learn to reproduce a XOR gate )

Congratulation! You built, trained and computed the output of your first Neural Network. 

You can try also the sample: sample_regression_sin, for regression tasks


2/ A more advanced classification (MNIST digits dataset)

The previous XOR sample is using the simple internal Matrix Library, but for heavy learning tasks, it is better to use the eigen library.
Eigen is a header only library, it compile out of the shelf with no dependencies.
BeeDNN can use either the internal matrix library or Eigen, change simply by setting a flag.

To use eigen:

1/Download eigen at : http://eigen.tuxfamily.org/
2/Unzip
3/Set the environmement variable "EIGEN_PATH" to the folder containing the unziped eigen, for example in my case: "C:\dev\eigen-3.4.0"
4/Reload Visual Studio 2019

Now, compile the sample sample_classification_MNIST (compile in Release, 64 bit mode, for good performances)

Before running this sample, download the MNIST train and test dataset,
Go to: http://yann.lecun.com/exdb/mnist/ , download and unzip the 4 .gz files (for example with 7-Zip (at: https://www.7-zip.org/), put the uzipped files (*.ubytes) near the sample executable.

Now, you can run sample_classification_MNIST, you should have an accuracy>98.1% after 15 epochs, 2s by epochs.
The learning speed and performances are comparable with Tensorflow on Dense layers.

3/ MNIST with convolution layers

Use the sample sample_classification_MNIST_all_convolutional, compiles and run
This sample is similar as the previous sample sample_classification_MNIST, but this network uses convolutional layers, the accuracy goes up to 99% and uses is 20s by epochs.

4/ Meta optimisation

Use the sample sample_MetaOptimizer_MNIST
Todo

5/ Contributing
Any help welcome!
For contributing idea and commits, please see the project github issues, or any other means.

Todo





