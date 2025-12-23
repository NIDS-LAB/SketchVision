#!/usr/bin/env bash

set -eux

ninja


ARR=(
    "bitcoinminer"
    "mazarbot"
    "ransombo"
    "wannalocker"
)

for item in ${ARR[@]}; do
    ./SketchVision -config ../configuration/c2/${item}.json > /tmp/${item}.log  &
done
