#!/bin/bash
N='1 3 5 7 9 12 16 20 30'

RES=res/res-lu-par-xeon-vector.txt
rm $RES

for n in $N
do
	export OMP_NUM_THREADS=244
	./lu-par r $((1024*$n)) 512 2>&1 >/dev/null | tee -a $RES
done

RES=res/res-lu-tile-xeon-vector.txt
rm $RES

for n in $N
do
	export OMP_NUM_THREADS=244
	./lu-tile r $((1024*$n)) 128 2>&1 >/dev/null | tee -a $RES
done

RES=res/res-lu-sca-par-xeon-vector.txt
rm $RES

for n in $N
do
	export OMP_NUM_THREADS=244
	./lu-sca-par r $((1024*$n)) 128 2>&1 >/dev/null | tee -a $RES
done

RES=res/res-lu-sca-tile-xeon-vector.txt
rm $RES

for n in $N
do
	export OMP_NUM_THREADS=244
	./lu-sca-tile r $((1024*$n)) 64 2>&1 >/dev/null | tee -a $RES
done

