#!/bin/bash

curl -O https://hailing-ling-public.s3-ap-northeast-1.amazonaws.com/GitHub/fascialsentiment/model/builtin_mobilenetv2-longrun.tar.bz2
tar jxvf builtin_mobilenetv2-longrun.tar.bz2
mv builtin_mobilenetv2-longrun/* ./
rm -rf builtin_mobilenetv2-longrun
rm -rf builtin_mobilenetv2-longrun.tar.bz2
