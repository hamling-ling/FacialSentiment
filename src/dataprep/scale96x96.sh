#!/bin/bash -x
find ../../data/input '*.png' | xargs mogrify -resize 96x96  -unsharp 10x5+0.7+0 -type GrayScale  -monitor
