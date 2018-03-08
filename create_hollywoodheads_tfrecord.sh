#!/bin/bash

# Specify data root and Syno mount point
DATA_ROOT="/data/hollywoodheads"
DATA_ZIP="${DATA_ROOT}/zips"

# Get data from MSCOCO
if [ ! -d "${DATA_ZIP}" ]; then
  echo "$(mkdir -p ${DATA_ZIP})" ;
fi  && \

URL_DATA="http://www.di.ens.fr/willow/research/headdetection/release/HollywoodHeads.zip"

wget "${URL_DATA}" -O "${DATA_ZIP}/data.zip" && \

# Construct arguments to pass to python
ARGS="--output_root=${DATA_ROOT} "
ARGS+="--num_val=8000 --train_shards=8 --val_shards=1 --test_shards=1 --num_threads=8"

# Unzip data located on Syno to local directories
echo "$(unzip ${DATA_ZIP}/data.zip -d ${DATA_ROOT})" && \

# Create TF Records
echo "$(python ./create_voc_tfrecord.py ${ARGS})"

# Remove temporary local directories
# echo "$(rm -rf ${DATA_ROOT}/zips)"