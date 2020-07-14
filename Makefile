ALL = samples

all: $(ALL)

clean:
	cd ./src/sample_code && make clean

samples:
	cd ./src/sample_code && make
	
test: all
	cd ./src/sample_code && make test
