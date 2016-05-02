#!/bin/bash
N=$((24*1024))

RES=res-lu-tile-xeon-24x.txt

rm $RES

for t in 61 122 244
do
	export OMP_NUM_THREADS=$t
	./lu-tile r $N 128 2>&1 >/dev/null | tee -a $RES
done

RES=res-lu-par-xeon-24x.txt

rm $RES

for t in 61 122 244
do
	export OMP_NUM_THREADS=$t
	./lu-par r $N 512 2>&1 >/dev/null | tee -a $RES
done


