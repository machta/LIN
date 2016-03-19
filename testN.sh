rm res.txt

for n in `seq 1 10` `seq 12 2 16` 20 32
do
	./lu-par r $((1024*$n)) 256 2>&1 >/dev/null | tee -a res.txt
done

