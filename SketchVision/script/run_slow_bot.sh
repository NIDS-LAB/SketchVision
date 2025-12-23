#!/usr/bin/env bash

set -eux

ninja

ARR=(
    "thbot"
    "telnetpwdmd"
    "telnetpwdla"
)

for item in ${ARR[@]}; do
    ./SketchVision -config ../configuration/bot/${item}.json > /tmp/${item}.log  &
done
