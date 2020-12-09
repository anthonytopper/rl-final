#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH -c 5
#SBATCH --gres=gpu:1
#SBATCH -C T4|K40|K80|V100
#SBATCH -t 24:00:00
#SBATCH --mem 22G
#SBATCH -p short
#SBATCH --job-name="prop"
source /home/aditthapron/work/[/home/aditthapron/work/anaconda3/bin/activate  sinc_2
echo "Raw Params:"
echo "$@"
python train.py "$@"
