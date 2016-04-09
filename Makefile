SHELL = /bin/bash
N = 10
K = 5
P = 2
FLAGS = -std=c++11 -Wall -pedantic -Ofast -march=native -D NDEBUG -fopenmp -fprofile-use $(CXXFLAGS)

SERIAL_JOB = serial_job.sh
PARALLEL_JOB = parallel_job.sh

TMP1 := $(shell mktemp)
TMP2 := $(shell mktemp)

COLUMN = lu-seq gauss-seq cholesky-seq
TILED  = lu-par
BLOCK_SIZES = 2 4 8 16 32 64 128 256 512 1024 10 20 30 40 50 100 500 1000
E = 0.000006 # Max allowed relative error.

BIN = $(COLUMN) $(TILED) error

all : $(BIN) 

debug :	CXXFLAGS=-U NDEBUG -O0 -g
debug : all

test : all
	for f in `find ./test -type f` ; do \
		for t in $(COLUMN) ; do \
			./$$t i < $$f > $(TMP2) 2>/dev/null ; \
			cat $$f | tail -n `cat $(TMP2) | wc -l` > $(TMP1) ; \
			[[ `cat $(TMP1) $(TMP2) | ./error 2>/dev/null` < $(E) ]] || echo Failed Test: "./$$t i < $$f" ; \
		done ; \
		\
		for t in $(TILED) ; do \
			for b in $(BLOCK_SIZES) ; do \
				./$$t i $$b < $$f > $(TMP2) 2>/dev/null ; \
				cat $$f | tail -n `cat $(TMP2) | wc -l` > $(TMP1) ; \
				[[ `cat $(TMP1) $(TMP2) | ./error 2>/dev/null` < $(E) ]] || echo Failed Test: "./$$t i $$b < $$f" ; \
			done ; \
		done ; \
	done
	
prof :
	rm -f *.gcda
	make clean
	make CXXFLAGS='$(CXXFLAGS) -fprofile-generate -fno-profile-use'
#	for b in $(COLUMN) ; do \
	for b in lu-seq gauss-seq ; do \
		export OMP_NUM_THREADS=1 ; ./$$b r 4096 256 >/dev/null ; \
	done
	export OMP_NUM_THREADS=1 ; ./lu-par r 8192 256 >/dev/null ;
	make clean
	
error : error.cpp
	$(CXX) -o $@ $^ $(FLAGS)
	
common.o : common.cpp common.h
	$(CXX) -c common.cpp $(FLAGS)
	
lu-seq : lu-seq.cpp common.o
	$(CXX) -o $@ $^ $(FLAGS)
	
lu-par : lu-par.cpp common.o
	$(CXX) -o $@ $^ $(FLAGS)
	
gauss-seq : gauss-seq.cpp common.o
	$(CXX) -o $@ $^ $(FLAGS)
	
cholesky-seq : cholesky-seq.cpp common.o
	$(CXX) -o $@ $^ $(FLAGS)
	
clean :
	rm -f $(BIN) error common.o

