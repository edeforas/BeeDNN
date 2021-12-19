BeeDNN is a C++ library and there is also an equivalent python, in this tutorial, we will focus on the C++ code 

Prerequisite:

A C++ compiler, this tutorial will use Visual Studio 2017, any recent version will be ok. This code can also be compiled under GCC or CLang

The sample codes are commented and should self explanatory, but do not hesitate to comment, ask or contribute.

1/First classification

For the first classification,  double clic on the all.sln solution file in the src/ folder, then:

Compile the sample sample_classification xor, run.
 You should see the text :
 
0_xor_0=0
0_xor_1=1
1_xor_0=1
1_xor_1=0
Test succeded.

( This sample learn to reproduce a XOR gate )

Congratulation! You built, trained and computed the output of you first Neural Network. 

You can try also the sample: sample_regression_sin, for regression tasks


2/ A more advanced classification (MNIST digits dataset)

The previous XOR sample is using the simple internal Matrix Library, but for heavy learning tasks, it is better to use the eigen library.
Eigen is a header only library, so it compile out of the shelf, under windows and linux, without other dependecies.
BeeDNN can use either the internal matrix library or Eigen, change simply by setting a flag.

To use eigen:

1/Download eigen at : http://eigen.tuxfamily.org/
2/Unzip
3/Set the environmement variable "EIGEN_PATH" to the folder containing the unziped eigen, for example in my case: "C:\dev\eigen-3.4.0"
4/Reload Visual Studio 2017

Now, you should be able to compile the sample sample_classification_MNIST (compile in Release, 64 bit mode, for good performances)

Before running the sample, you need to download the MNIST train and test dataset, (this dataset is used very frequently)
Go to: http://yann.lecun.com/exdb/mnist/ , download and unzip the 4 .gz files ( 7-Zip (at : https://www.7-zip.org/ ) can be used to do the unzip taks, put the uzipped files (*.ubytes) near the sample executable.

Now, you can run sample_classification_MNIST, you should have an accuracy>98.1% after 15 epochs, 2s by epochs.
The learning speed and performances are comparable with Tensorflow on Dense layers.

3/ MNIST with convolution layers

Use the sample sample_classification_MNIST_all_convolutional, compiles and run
Todo

4/ Meta optimisation

Use the sample sample_MetaOptimizer_MNIST
Todo

5/ Contributing
Any help welcome!
For contributing idea and commits, please see the project github issues, or do the best way for you.

Todo





