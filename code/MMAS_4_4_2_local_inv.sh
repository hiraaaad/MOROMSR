#!/bin/bash 
#SBATCH --array=1-50 
#SBATCH -p batch 
#SBATCH --output /dev/null 
#SBATCH -N 1            # number of nodes 
#SBATCH -n 1            # number of cores 
#SBATCH --time=1-12:00:00#SBATCH --mem=500MB      # memory pool for all cores 
#SBATCH --mail-type=END 
#SBATCH --mail-user=hirad.assimi@adelaide.edu.au 
module load Anaconda3 
id=0 
rep1=0 
while [ $rep1 -le 0 ] 
do 
rep2=1 
while [ $rep2 -le 1 ] 
do 
rep3=1 
while [ $rep3 -le 51 ] 
do 
if [ $id -eq $SLURM_ARRAY_TASK_ID ] 
then 
echo $id 
source activate eka 
python -u run_phoenix_mmas.py 4 4 10 4 0 1 3 $id 2 1 2 0.5 10 1000 
conda deactivate 
fi 
let id++ 
let rep3++
done 
let rep2++
done 
let rep1++
done 
