#!/bin/bash
N=$((16*1024))

RES=res-lu-tile-xeon-block.txt
./lu-tile r $N 16 >/dev/null 2>$RES
./lu-tile r $N 32 >/dev/null 2>>$RES
./lu-tile r $N 64 >/dev/null 2>>$RES
./lu-tile r $N 128 >/dev/null 2>>$RES 
./lu-tile r $N 256 >/dev/null 2>>$RES
./lu-tile r $N 512 >/dev/null 2>>$RES

RES=res-lu-par-xeon-block.txt
./lu-par r $N 16 >/dev/null 2>$RES
./lu-par r $N 32 >/dev/null 2>>$RES
./lu-par r $N 64 >/dev/null 2>>$RES
./lu-par r $N 128 >/dev/null 2>>$RES 
./lu-par r $N 256 >/dev/null 2>>$RES
./lu-par r $N 512 >/dev/null 2>>$RES

