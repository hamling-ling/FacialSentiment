#!/bin/bash -x
FIND_PATH=../../data/input
find $FIND_PATH -name '*.png' | xargs mogrify -resize 96x96 -unsharp 10x5+0.7+0 -type GrayScale -format jpg -monitor 

echo "deleting"
find $FIND_PATH -name '*.png' | xargs rm -rf
