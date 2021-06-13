ALL = samples

all: $(ALL)

clean:
	cd ./samples && make clean

samples:
	cd ./samples && make
	
test: all
	cd ./samples && make test
