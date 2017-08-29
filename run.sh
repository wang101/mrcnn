#!/bin/csh
module load tensorflow/1.0-python2
python train_frcnn.py -o simple -p multilabel.txt

