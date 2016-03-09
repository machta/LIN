SHELL = /bin/bash
N = 10
K = 5
P = 2
FLAGS = -std=c++11 -Wall -pedantic -Ofast -march=native -D NDEBUG $(CXXFLAGS)
SERIAL_JOB = serial_job.sh
PARALLEL_JOB = parallel_job.sh

TEST = 	[ `./hon-seq < in$(1).txt 2>/dev/null | head -1` == $(2) ] &&\
		[ `mpirun -np 1  ./hon-par < in$(1).txt 2>/dev/null | head -1` == $(2) ] &&\
		[ `mpirun -np 3  ./hon-par < in$(1).txt 2>/dev/null | head -1` == $(2) ] &&\
		[ `mpirun -np 4  ./hon-par < in$(1).txt 2>/dev/null | head -1` == $(2) ] &&\
		[ `mpirun -np 16 ./hon-par < in$(1).txt 2>/dev/null | head -1` == $(2) ]

all : lu-seq

debug :	CXXFLAGS=-U NDEBUG -O0 -g
debug : all

test : all
	$(call TEST,0,3)
	$(call TEST,1,3)
	$(call TEST,2,3)
	$(call TEST,3,3)
	$(call TEST,4,1)
	
common.o : common.cpp common.h
	$(CXX) -c common.cpp $(FLAGS)
	
lu-seq : lu-seq.cpp common.o
	$(CXX) -o $@ $^ $(FLAGS)
	
clean :
	rm -f lu-seq common.o

