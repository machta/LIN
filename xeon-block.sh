#!/bin/bash
N=$((16*1024))

RES=res-lu-tile-xeon-block.txt
rm $RES

for b in 16 32 64 128 256 512
do
	./lu-tile r $N $b 2>&1 >/dev/null | tee -a $RES
done

RES=res-lu-par-xeon-block.txt
rm $RES

for b in 16 32 64 128 256 512 1024 2048
do
        ./lu-par r $N $b 2>&1 >/dev/null | tee -a $RES
done

RES=res-lu-sca-tile-xeon-block.txt
rm $RES

for b in 16 32 64 128 256 512 
do
	./lu-sca-tile r $N $b 2>&1 >/dev/null | tee -a $RES
done

RES=res-lu-sca-par-xeon-block.txt
rm $RES

for b in 16 32 64 128 256 512 1024 2048
do
	./lu-sca-par r $N $b 2>&1 >/dev/null | tee -a $RES
done
