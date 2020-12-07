#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH -c 5
#SBATCH --gres=gpu:1
#SBATCH -C T4|K40|K80
#SBATCH -t 72:00:00
#SBATCH --mem 50G
#SBATCH -p long
#SBATCH --job-name="prop"
source /home/aditthapron/work/[/home/aditthapron/work/anaconda3/bin/activate  sinc
echo "Raw Params:"
echo "$@"
python main.py --train_dqn "$@"
