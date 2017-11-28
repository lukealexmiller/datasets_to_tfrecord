#!/bin/sh
echo "Cloning cocoapi into /root/scripts" && \
cd /root/scripts && \
git clone https://github.com/cocodataset/cocoapi && \
cd  cocoapi/PythonAPI/ && \
pip install Cython && \
make && \
echo "Build complete"