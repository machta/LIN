#!/bin/bash
N=$((16*1024))

for t in 61 122 244
do
	for p in lu-tile lu-par lu-sca-tile lu-sca-par
	do
		RES="res/res-$p-xeon-block-"$t"x.txt"
		rm $RES
		
		for b in 16 32 64 128 256 512 1024 2048
		do
			export OMP_NUM_THREADS=$t
			./$p r $N $b 2>&1 >/dev/null | tee -a $RES
		done
	done
done
