#!/usr/bin/env bash

set -eux

ninja


ARR=(
    "mazarbot"
)

for item in ${ARR[@]}; do
    ./SketchVision -config ../configuration/c2/${item}.json > /tmp/${item}.log  &
done
