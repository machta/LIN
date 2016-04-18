SHELL = /bin/bash
FLAGS = -std=c++11 -Wall -pedantic -Ofast -march=native -D NDEBUG -fopenmp -fprofile-use $(CXXFLAGS)
TMP1 := $(shell mktemp)
TMP2 := $(shell mktemp)
SIMPLE = lu-seq lu-sca-seq
BLOCK  = lu-par lu-tile lu-sca-par lu-sca-tile
BLOCK_SIZES = 2 4 8 16 32 64 128 256 512 1024 10 20 30 40 50 100 500 1000
E = 0.000006 # Max allowed relative error.
BIN = $(SIMPLE) $(BLOCK) error

all : $(BIN) 

debug :	CXXFLAGS=-U NDEBUG -O0 -g
debug : all

test : all
	for f in `find ./test -type f` ; do \
		for t in $(SIMPLE) ; do \
			./$$t i < $$f > $(TMP2) 2>/dev/null ; \
			cat $$f | tail -n `cat $(TMP2) | wc -l` > $(TMP1) ; \
			[[ `cat $(TMP1) $(TMP2) | ./error 2>/dev/null` < $(E) ]] || echo Failed Test: "./$$t i < $$f" ; \
		done ; \
		\
		for t in $(BLOCK) ; do \
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
	for t in $(BLOCK) ; do \
		for b in 64 128 256 ; do \
			for n in 1024 2048 4096 ; do \
				export OMP_NUM_THREADS=1 ; ./$$t r $$n $$b >/dev/null ; \
			done ; \
		done ; \
	done ; \
	make clean

.PHONY:jobs
jobs :
	ls jobs/*.sh | xargs -n1 qsub 

gnuplot :
	gnuplot gnuplot-speedup.txt gnuplot-block.txt gnuplot-sequential.txt
	gnuplot gnuplot-speedup-phi.txt #gnuplot-block-phi.txt
	
error : error.cpp
	$(CXX) -o $@ $^ $(FLAGS)
	
common.o : common.cpp common.h
	$(CXX) -c common.cpp $(FLAGS)
	
lu-seq : lu-seq.cpp common.o
	$(CXX) -o $@ $^ $(FLAGS)
	
lu-par : lu-par.cpp common.o
	$(CXX) -o $@ $^ $(FLAGS)

lu-tile : lu-tile.cpp common.o
	$(CXX) -o $@ $^ $(FLAGS)
	
lu-sca-seq : lu-seq.cpp common.o
	$(CXX) -o $@ $^ $(FLAGS) -fno-tree-vectorize
	
lu-sca-par : lu-par.cpp common.o
	$(CXX) -o $@ $^ $(FLAGS) -fno-tree-vectorize

lu-sca-tile : lu-tile.cpp common.o
	$(CXX) -o $@ $^ $(FLAGS) -fno-tree-vectorize

clean :
	rm -f $(BIN) error common.o *.jpg

