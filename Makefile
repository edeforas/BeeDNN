ALL = samples

all: $(ALL)

clean:
	cd ./sample_code && make clean

samples:
	cd ./sample_code && make
	
test: all
	cd ./sample_code && make test
