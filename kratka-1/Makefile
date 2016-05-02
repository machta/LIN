FLAGS = -std=c++11 -Wall -pedantic -Ofast -march=native -D NDEBUG -fopenmp -fprofile-use $(CXXFLAGS) -Wno-unused-variable

.PHONY : all
all : main

.PHONY : prof
prof :
	rm -f *.gcda
	make clean
	make CXXFLAGS='$(CXXFLAGS) -fprofile-generate -fno-profile-use'
	export OMP_NUM_THREADS=1 ; ./main
	make clean
	
.PHONY : points
points : main
	echo 0.324/`./main | egrep 'Total time' | sed -r 's/.*=(.*)/\1/'`*9 | bc -l

main : main.cpp
	$(CXX) -o $@ $^ $(FLAGS)

.PHONY : clean
clean :
	rm -f main
