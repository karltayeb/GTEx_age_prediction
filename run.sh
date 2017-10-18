#!/bin/bash -l
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=4G

module load anaconda-python/2.7
pip install --user seaborn

MAT_DIR=/home-3/ktayeb1@jhu.edu/work/GTEx_v6/GTEx_Analysis_v6p_eQTL_expression_matrices 
ANNOT_DIR=/home-3/ktayeb1@jhu.edu/work/GTEx_v6

KRANGE=10
PRANGE=10
KFOLDS=10
RUNS=20

python split_data.py $MAT_DIR $ANNOT_DIR
python tune.py 0 $KRANGE $PRANGE $KFOLDS
python tune.py 1 $KRANGE $PRANGE $KFOLDS
python tune.py 2 $KRANGE $PRANGE $KFOLDS
python tune.py 3 $KRANGE $PRANGE $KFOLDS

python learning_curves.py $RUNS
python gen_plots.py

