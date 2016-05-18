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

P=lu-cu-unroll
export CUDA_VISIBLE_DEVICES=1

for n in 8 16
do
	RES=res-$P-GPU2$CUDA_VISIBLE_DEVICES-block-"$n"x.txt
	rm $RES
	
	for t in 1 2 4 8 16 32 
	do
		./$P r $((1024*$n)) $t 2>&1 >/dev/null | tee -a $RES
	done
done

