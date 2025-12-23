#!/usr/bin/env bash

set -eux

BASE_NAME=$(basename $(pwd))

if [ $BASE_NAME != "SketchVision" ]; then
    echo "This script should be executed in the root dir of SketchVision."
    exit -1
fi

echo "Compiling SketchVision."

if [ -d "./build" ]; then
    cd build && ninja 
else
    echo "Build the project first using cmake."
fi

