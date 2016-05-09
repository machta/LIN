SHELL = /bin/bash
FLAGS = -std=c++11 -D NDEBUG $(CXXFLAGS)
CFLAGS = -pedantic -Wall -Ofast -march=native -fopenmp -fprofile-use $(FLAGS)
NVFLAGS = -O3 -lcublas $(FLAGS)
TMP1 := $(shell mktemp)
TMP2 := $(shell mktemp)
SIMPLE = lu-seq lu-sca-seq lu-cu-blas
BLOCK  = lu-par lu-tile lu-sca-par lu-sca-tile
BIN = $(SIMPLE) $(BLOCK) error
BLOCK_SIZES = 2 4 8 16 32 64 128 256 512 1024 10 20 30 40 50 100 500 1000
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
		for t in $(BLOCK) ; do \
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
	make CXXFLAGS='$(CXXFLAGS) -fprofile-generate -fno-profile-use'
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
	ls gnuplot-*.txt | xargs -n1 gnuplot
	
.PHONY : xeon
xeon :
	make CXX=icc CXXFLAGS='-mmic -vec-report'

error : error.cpp
	$(CXX) -o $@ $^ $(CFLAGS)
	
common.o : common.cpp common.h
	$(CXX) -c common.cpp $(CFLAGS)
	
lu-seq : lu-seq.cpp common.o
	$(CXX) -o $@ $^ $(CFLAGS)
	
lu-par : lu-par.cpp common.o
	$(CXX) -o $@ $^ $(CFLAGS)

lu-tile : lu-tile.cpp common.o
	$(CXX) -o $@ $^ $(CFLAGS)
	
lu-sca-seq : lu-seq.cpp common.o
	$(CXX) -o $@ $^ $(CFLAGS) -fno-tree-vectorize
	
lu-sca-par : lu-par.cpp common.o
	$(CXX) -o $@ $^ $(CFLAGS) -fno-tree-vectorize

lu-sca-tile : lu-tile.cpp common.o
	$(CXX) -o $@ $^ $(CFLAGS) -fno-tree-vectorize

lu-cu-blas : lu-cu-blas.cu common.o
	nvcc -o $@ $^ $(NVFLAGS)

.PHONY : clean
clean :
	rm -f $(BIN) error common.o *.jpg
	rm -fr graphs

