#!/usr/bin/env bash

set -eux

ninja

ARR=(
    "adload"
    "mobidash"
    "webcompanion"
)

for item in ${ARR[@]}; do
    ./SketchVision -config ../configuration/surveil/${item}.json > /tmp/${item}.log  &
done
