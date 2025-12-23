#!/usr/bin/env bash

set -eux

BASE_NAME=$(basename $(pwd))

if [ $BASE_NAME != "SketchVision" ]; then
    echo "This script should be executed in the root dir of SketchVision."
    exit -1
fi

echo "Rebuild SketchVision."

if [ -d "./build" ]; then
    echo "Old build dir is removed."
    rm -r ./build
fi

mkdir build && cd $_ && cmake -G Ninja .. && cd ..

if [ -f "compile_commands.json" ]; then
    rm compile_commands.json
    ln -s build/compile_commands.json .
else
    ln -s build/compile_commands.json .
fi

if [ $? == 0 ]; then
    echo "Rebuild finished."
else
    echo "Rebuild failed."
fi
