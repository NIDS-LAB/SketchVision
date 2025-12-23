#!/usr/bin/env bash

set -eux

ninja

ARR=(
    "dridex"
    "feiwo"
    "snojan"
    "penetho"
    "koler"
)

for item in ${ARR[@]}; do
    ./SketchVision -config ../configuration/exfil/${item}.json > /tmp/${item}.log  &
done
