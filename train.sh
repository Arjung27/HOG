#!/bin/sh

#SBATCH --partition scavenger

#SBATCH --gres=gpu:4

#SBATCH --ntasks=16

#SBATCH --mem=128G

#SBATCH --account=scavenger

#SBATCH --qos=scavenger

#SBATCH --time=72:00:00

#SBATCH --output=cls-test.out

#SBATCH --job-name=cls_test

source /cmlscratch/${USER}/anaconda3/etc/profile.d/conda.sh
conda activate python37

export PYTHONPATH=./
#eval "$(conda shell.bash hook)"
#conda activate pt140  # pytorch 1.4.0 env
PYTHON=python

now=$(date +"%Y%m%d_%H%M%S")

$PYTHON -u ./train.py /fs/cml-datasets/ImageNet/ILSVRC2012 \
  --arch vgg11_H1 -j 16 -b 256 --lr 0.01 \
  --world-size 1 --rank 0 --dist-url tcp://localhost:17456 --multiprocessing-distributed \
  --hog_ppc 3 --hog_cpb 1 --stacked_dims \
  2>&1 | tee ./train-$now.log

