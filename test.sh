#!/bin/csh
module load tensorflow/1.0-python2
python test_frcnn.py -p ./data/img_test
