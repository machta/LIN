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
#$ -q gpu_long.q

P=lu-sca-par
T=24

RES="res/res-$P-block-"$T"x.txt"
N=$((8*1024))

rm $RES

for b in 8 16 `seq 32 32 256` `seq $((256+64)) 64 1024` 2048
do
	export OMP_NUM_THREADS=$T
	./$P r $N $b 2>&1 >/dev/null | tee -a $RES
done

