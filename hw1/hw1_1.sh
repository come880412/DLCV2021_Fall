#!/bin/bash
wget 'https://www.dropbox.com/s/vixbgqgjw8ihl8h/model_epoch9_acc89.60.pth?dl=1'
python3 ./code/p1/test.py $1 $2