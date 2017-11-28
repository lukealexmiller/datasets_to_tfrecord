#!/bin/sh
echo "Cloning cocoapi into /root/data/scripts" && \
cd /root/data/scripts && \
git clone https://github.com/cocodataset/cocoapi && \
cd  cocoapi/PythonAPI/ && \
pip install Cython && \
make && \
echo "Build complete"