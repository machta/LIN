SHELL = /bin/bash

FLAGS = -std=c++11 -D NDEBUG $(CXXFLAGS)
CFLAGS = -pedantic -Wall -Ofast -march=native -fopenmp -fprofile-use $(FLAGS)
NVFLAGS = -O3 $(FLAGS) -D_MWAITXINTRIN_H_INCLUDED -D_FORCE_INLINES #-D__STRICT_ANSI__

TMP1 := $(shell mktemp)
TMP2 := $(shell mktemp)

SIMPLE = lu-seq lu-sca-seq
BLOCK = lu-par lu-tile lu-sca-par lu-sca-tile
CUDA = lu-cu-sdk lu-cu-simple lu-cu-unroll lu-cu-memory
BIN = $(SIMPLE) $(BLOCK) $(CUDA) error

BLOCK_SIZES = 1 2 3 4 5 8 10 12 16 24 32 64 128 256 512 1024
E = 0.000006 # Max allowed relative error.

.PHONY : all
all : $(BIN)

.PHONY : debug
debug :	CXXFLAGS=-U NDEBUG -O0 -g
debug : all

.PHONY : test
test : all
	for f in `find ./test -type f` ; do \
		for t in $(SIMPLE) ; do \
			./$$t i < $$f > $(TMP2) 2>/dev/null ; \
			cat $$f | tail -n `cat $(TMP2) | wc -l` > $(TMP1) ; \
			[[ `cat $(TMP1) $(TMP2) | ./error 2>/dev/null` < $(E) ]] || echo Failed Test: "./$$t i < $$f" ; \
		done ; \
		\
		for t in $(BLOCK) $(CUDA) ; do \
			for b in $(BLOCK_SIZES) ; do \
				./$$t i $$b < $$f > $(TMP2) 2>/dev/null ; \
				cat $$f | tail -n `cat $(TMP2) | wc -l` > $(TMP1) ; \
				[[ `cat $(TMP1) $(TMP2) | ./error 2>/dev/null` < $(E) ]] || echo Failed Test: "./$$t i $$b < $$f" ; \
			done ; \
		done ; \
	done
	
.PHONY : prof
prof :
	rm -f *.gcda
	make clean
	make $(BLOCK) CXXFLAGS='$(CXXFLAGS) -fprofile-generate -fno-profile-use'
	for t in $(BLOCK) ; do \
		for b in 64 128 256 ; do \
			for n in 1024 2048 4096 ; do \
				export OMP_NUM_THREADS=1 ; ./$$t r $$n $$b >/dev/null ; \
			done ; \
		done ; \
	done ; \
	make clean

.PHONY : jobs
jobs :
	ls jobs/*.sh | xargs -n1 qsub 

.PHONY : gnuplot
gnuplot :
	mkdir -p graphs
	mkdir -p graphs/x86
	mkdir -p graphs/xeon
	mkdir -p graphs/cuda
	ls gnuplot-*.txt | xargs -n1 gnuplot
	
.PHONY : xeon
xeon :
	make $(SIMPLE) $(BLOCK) CXX=icc CXXFLAGS='-mmic -vec-report'

error : src/error.cpp
	$(CXX) -o $@ $^ $(CFLAGS) -fno-profile-use
	
common.o : src/common.cpp src/common.h
	$(CXX) -c src/common.cpp $(CFLAGS)
	
lu-seq : src/lu-seq.cpp common.o
	$(CXX) -o $@ $^ $(CFLAGS) -fno-profile-use
	
lu-par : src/lu-par.cpp common.o
	$(CXX) -o $@ $^ $(CFLAGS)

lu-tile : src/lu-tile.cpp common.o
	$(CXX) -o $@ $^ $(CFLAGS)
	
lu-sca-seq : src/lu-seq.cpp common.o
	$(CXX) -o $@ $^ $(CFLAGS) -fno-tree-vectorize -fno-profile-use
	
lu-sca-par : src/lu-par.cpp common.o
	$(CXX) -o $@ $^ $(CFLAGS) -fno-tree-vectorize

lu-sca-tile : src/lu-tile.cpp common.o
	$(CXX) -o $@ $^ $(CFLAGS) -fno-tree-vectorize

lu-cu-sdk : src/lu-cu-sdk.cu common.o
	nvcc -o $@ $^ -lcusolver $(NVFLAGS)
	
lu-cu-simple : src/lu-cu-simple.cu common.o
	nvcc -o $@ $^ $(NVFLAGS)
	
lu-cu-unroll : src/lu-cu-unroll.cu common.o
	nvcc -o $@ $^ $(NVFLAGS)
	
lu-cu-memory : src/lu-cu-memory.cu common.o
	nvcc -o $@ $^ $(NVFLAGS)

.PHONY : clean
clean :
	rm -f $(BIN) error common.o *.jpg
	rm -fr graphs

