rm res.txt

for b in 8 16 `seq 32 32 256` `seq $((256+64)) 64 1024` 2048
do
	./lu-par r $((1024*8)) $b 2>&1 >/dev/null | tee -a res.txt
done


