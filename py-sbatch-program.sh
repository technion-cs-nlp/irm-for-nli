#!/bin/bash

###
# CS236781: Deep Learning
# py-sbatch.sh
#
# This script runs python from within our conda env as a slurm batch job.
# All arguments passed to this script are passed directly to the python
# interpreter.
#

###
# Example usage:
#
# Running the prepare-submission command from main.py as a batch job
# ./py-sbatch.sh main.py prepare-submission --id 123456789
#
# Running all notebooks without preparing a submission
# ./py-sbatch.sh main.py run-nb *.ipynb
#
# Running any other python script myscript.py with arguments
# ./py-sbatch.sh myscript.py --arg1 --arg2=val2
#

#
####
## Parameters for sbatch
##
NUM_NODES=1
NUM_CORES=2
NUM_GPUS=1
JOB_NAME="irm-repr"
MAIL_USER="yanadr@campus.technion.ac.il"
MAIL_TYPE=ALL # Valid values are NONE, BEGIN, END, FAIL, REQUEUE, ALL

###
# Conda parameters
#
CONDA_HOME=$HOME/miniconda3
CONDA_ENV=irm_for_nli


echo "*** SLURM BATCH JOB '$JOB_NAME' STARTING ***"
# Setup the conda env
echo "*** Activating environment $CONDA_ENV ***"
source $CONDA_HOME/etc/profile.d/conda.sh
conda activate $CONDA_ENV
# Run python with the args to the script
DIR=$1
FILE=$2

sbatch \
        -N $NUM_NODES \
        -c $NUM_CORES \
        --gres=gpu:$NUM_GPUS \
        --job-name $JOB_NAME \
        --mail-user $MAIL_USER \
        --mail-type $MAIL_TYPE \
        -o 'slurm-%N-%j.out' \
	-p 'nlp' \
<<EOF
#!/bin/bash

###
# setup
if [ ! -d "$DIR" ]; then
	mkdir "$DIR"
	touch "$DIR/log.txt"
	cp "$FILE" "$DIR/run_commands.txt"
fi

n=0
echo "before loop \$n"
while read line;
do 
	if [ "\$line" == "" ]; then
		break
	else
		n=\$((n+1))
	fi
done < "$DIR/log.txt"
echo "after loop: \$n"

j=1
while read line;
do
	if [ "\$line" == "" ]; then
		echo "empty row \$j"
		echo "\$line"
		break
	elif [ \$j -gt \$n ]; then
		python \$line
		echo "\$line"
		echo "finished command  \$line" >> "$DIR/log.txt"
	fi
	j=\$((j+1))
done < "$DIR/run_commands.txt"
EOF

