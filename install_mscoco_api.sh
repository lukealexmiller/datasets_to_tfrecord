#!/bin/sh
echo "Cloning cocoapi into /data/scripts" && \
cd /data/scripts && \
git clone https://github.com/cocodataset/cocoapi && \
cd  cocoapi/PythonAPI/ && \
pip install Cython && \
make && \
echo "Build complete"