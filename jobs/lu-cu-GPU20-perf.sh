#!/bin/sh

#  ===========================================================================
# |                                                                           |
# |             COMMAND FILE FOR SUBMITTING SGE JOBS                          |
# |                                                                           |
# |                                                                           |
# | SGE keyword statements begin with #$                                      |
# |                                                                           |
# | Comments begin with #                                                     |
# | Any line whose first non-blank character is a pound sign (#)              |
# | and is not a SGE keyword statement is regarded as a comment.              |
#  ===========================================================================

# Request Bourne shell as shell for job
#$ -S /bin/sh

# Execute the job from the current working directory.
#$ -cwd

# Defines  or  redefines  the  path used for the standard error stream of the job.
#$ -e .

# The path used for the standard output stream of the job.
#$ -o .

# Do not change.
#$ -pe ompi 1

# Do not change.
#$ -q gpu_02.q

export CUDA_VISIBLE_DEVICES=0

for n in `seq 1 10` `seq 12 2 16` 20 24 28
do
	P=lu-cu-sdk
	RES=res-$P-GPU2$CUDA_VISIBLE_DEVICES-perf.txt
        if [ $n == "1" ]
        then        
	        rm $RES
        fi
	./$P r $((1024*$n)) 2>&1 >/dev/null | tee -a $RES
	
	P=lu-cu-unroll
	RES=res-$P-GPU2$CUDA_VISIBLE_DEVICES-perf.txt
	if [ $n == "1" ]
        then        
	        rm $RES
        fi
	./$P r $((1024*$n)) 4 2>&1 >/dev/null | tee -a $RES
	
	P=lu-cu-simple
	RES=res-$P-GPU2$CUDA_VISIBLE_DEVICES-perf.txt
        if [ $n == "1" ]
        then        
		rm $RES
	fi
	./$P r $((1024*$n)) 2>&1 >/dev/null | tee -a $RES
done
