CXXFLAGS = -g -Wall -Wfatal-errors -std=c++17

ALL = sample_regression_sin sample_classification_xor

all: $(ALL)

sample_regression_sin: sample_regression_sin.cpp Makefile
	$(CXX) $(CXXFLAGS) sample_regression_sin.cpp -I.. ../*.cpp -o sample_regression_sin -lm -lpthread

sample_classification_xor: sample_classification_xor.cpp Makefile
	$(CXX) $(CXXFLAGS) sample_classification_xor.cpp -I.. ../*.cpp -o sample_classification_xor -lm -lpthread

clean:
	$(RM) $(ALL) *.o

test: all
	./sample_regression_sin && ./sample_classification_xor
