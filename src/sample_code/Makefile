CXXFLAGS = -g -Wall -Wfatal-errors -std=c++14

ALL = sample_regression_sin

all: $(ALL)

sample_regression_sin: sample_regression_sin.cpp Makefile
	$(CXX) $(CXXFLAGS) sample_regression_sin.cpp -I.. ../*.cpp -o sample_regression_sin -lm -lpthread

clean:
	$(RM) $(ALL) *.o

test: all
	./sample_regression_sin
