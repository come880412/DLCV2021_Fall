#!/bin/bash
wget 'https://www.dropbox.com/s/ff24e63hao3c7bh/model_miou71.54.pth?dl=1'
python3 ./code/p2/test.py $1 $2