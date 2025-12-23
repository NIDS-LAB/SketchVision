#!/bin/bash

# Directories to remove
DIRS=("Result" "diffusion-training-images" "diffusion-videos" "model")

for d in "${DIRS[@]}"; do
    if [ -d "$d" ]; then
        echo "Removing directory: $d"
        rm -rf "$d"
    else
        echo "Directory not found: $d"
    fi
done

echo "Done."
