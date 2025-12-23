#!/usr/bin/env bash

set -eux

ninja

ARR=(
    "ssdprdos"
    "cldaprdos"
    "riprdos"
    "charrdos"
)

for item in ${ARR[@]}; do
    ./SketchVision -config ../configuration/ddos/${item}.json > /tmp/${item}.log  &
done
