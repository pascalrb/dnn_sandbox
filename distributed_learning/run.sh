#!/bin/bash

for i in {1..24};
do 
    echo -e '\n'${i}' process:'
    time python3 distrib-lenet5-vgg.py --num_workers=$i >> res.txt
done

echo -e '\n32 process:'
time python3 distrib-lenet5-vgg.py --num_workers=32 >> res.txt
