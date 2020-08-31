#CXXFLAGS = -g -Wall -Wfatal-errors -std=c++17 -fopenmp -I/eigenfolder -DUSE_EIGEN -O3
CXXFLAGS = -g -Wall -Wfatal-errors -std=c++17 -fopenmp -O3

ALL = sample_regression_sin sample_classification_xor sample_classification_MNIST sample_classification_MNIST_allconvolutional sample_classification_MNIST_poolmax2d sample_MetaOptimizer_MNIST

all: $(ALL)

sample_regression_sin: sample_regression_sin.cpp Makefile
	$(CXX) $(CXXFLAGS) sample_regression_sin.cpp -I.. ../*.cpp -o sample_regression_sin -lm -lpthread

sample_classification_xor: sample_classification_xor.cpp Makefile
	$(CXX) $(CXXFLAGS) sample_classification_xor.cpp -I.. ../*.cpp -o sample_classification_xor -lm -lpthread

sample_classification_MNIST: sample_classification_MNIST.cpp Makefile
	$(CXX) $(CXXFLAGS) sample_classification_MNIST.cpp -I.. ../*.cpp -o sample_classification_MNIST -lm -pthread

sample_classification_MNIST_allconvolutional: sample_classification_MNIST_allconvolutional.cpp Makefile
	$(CXX) $(CXXFLAGS) sample_classification_MNIST_allconvolutional.cpp -I.. ../*.cpp -o sample_classification_MNIST_allconvolutional -lm -pthread

sample_classification_MNIST_poolmax2d: sample_classification_MNIST_poolmax2d.cpp Makefile
	$(CXX) $(CXXFLAGS) sample_classification_MNIST_poolmax2d.cpp -I.. ../*.cpp -o sample_classification_MNIST_poolmax2d -lm -pthread

sample_MetaOptimizer_MNIST: sample_MetaOptimizer_MNIST.cpp Makefile
	$(CXX) $(CXXFLAGS) sample_MetaOptimizer_MNIST.cpp -I.. ../*.cpp -o sample_MetaOptimizer_MNIST -lm -pthread

clean:
	$(RM) $(ALL) *.o

test: all
	./sample_regression_sin && ./sample_classification_xor
