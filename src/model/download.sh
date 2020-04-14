#!/bin/bash

curl -O https://hailing-ling-public.s3-ap-northeast-1.amazonaws.com/GitHub/fascialsentiment/model/model.tar.bz2
tar jxvf model.tar.bz2
mv model/* ./
rm -rf model
rm -rf modeltar.bz2
