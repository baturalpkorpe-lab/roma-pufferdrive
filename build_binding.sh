#!/bin/bash
# Run from ~/PufferDrive after cloning.
# Compiles the C binding and generates map binaries.
set -e

echo "[build] Creating pufferlib __init__.py files..."
touch pufferlib/ocean/__init__.py
touch pufferlib/ocean/drive/__init__.py

echo "[build] Compiling binding..."
NUMPY_INC=$(python3 -c "import numpy; print(numpy.get_include())")
gcc -O2 -shared -fPIC \
    -I/usr/include/python3.11 -I"$NUMPY_INC" -I. \
    -I./raylib-5.5_linux_amd64/include \
    -I./box2d-linux-amd64/include \
    -I./inih-r62 \
    pufferlib/ocean/drive/binding.c ./inih-r62/ini.c \
    ./raylib-5.5_linux_amd64/lib/libraylib.a \
    ./box2d-linux-amd64/libbox2d.a \
    -o pufferlib/ocean/drive/binding.cpython-311-x86_64-linux-gnu.so \
    -lm -lpthread -ldl

echo "[build] Generating map binaries..."
PYTHONPATH=. python3 -c "
from pufferlib.ocean.drive.drive import process_all_maps
process_all_maps(data_folder='data/processed/training', num_workers=8)
"
echo "[build] Done."
