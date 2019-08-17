CXXFLAGS = -g -Wall -Wfatal-errors -std=c++17

ALL = samples

all: $(ALL)

clean:
	$(RM) $(ALL) *.o

samples:
	cd ./src/sample_code && make
	
test: all
	cd ./src/sample_code && make test
