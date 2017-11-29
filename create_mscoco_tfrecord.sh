#!/bin/bash

# Specify data root and Syno mount point
DATA_ROOT="/root/data/mscoco"
DATA_ZIP="${DATA_ROOT}/zips"

# Specify MSCOCO dataset year
YEAR="2014"

# Get data from MSCOCO
if [ ! -d "${DATA_ZIP}" ]; then 
  echo "$(mkdir -p ${DATA_ZIP})" ;
fi  && \

URL_TRAIN="http://images.cocodataset.org/zips/train${YEAR}.zip"
URL_VAL="http://images.cocodataset.org/zips/val${YEAR}.zip"
URL_ANNO="http://images.cocodataset.org/annotations/annotations_trainval${YEAR}.zip"

#wget "${URL_TRAIN}" -O "${DATA_ZIP}/train${YEAR}.zip" && \
#wget "${URL_VAL}" -O "${DATA_ZIP}/val${YEAR}.zip" && \
#wget "${URL_ANNO}" -O "${DATA_ZIP}/annotations_trainval${YEAR}.zip" && \

# Construct arguments to pass to python
ARGS="--data_root=${DATA_ROOT} --output_root=${DATA_ROOT} --year=${YEAR} "
ARGS+="--shuffle=True --num_val=8000 --train_shards=64 --val_shards=4 --test_shards=0 --num_threads=8"

# Install MSCOCO API
if [ ! -d "/root/data/scripts/cocoapi" ]; then 
  echo "$(chmod +x install_mscoco_api.sh)" && \
  ./install_mscoco_api.sh ; 
fi  && \

# Unzip data located on Syno to local directories
echo "$(unzip ${DATA_ZIP}/val${YEAR}.zip -d ${DATA_ROOT})" && \
echo "$(unzip ${DATA_ZIP}/train${YEAR}.zip -d ${DATA_ROOT})" && \
echo "$(unzip ${DATA_ZIP}/annotations_trainval${YEAR}.zip -d ${DATA_ROOT})" && \

# Create TF Records
echo "$(python ./create_mscoco_tfrecord.py ${ARGS})"

# Remove temporary local directories
echo "$(rm -rf ${DATA_ROOT}/train${YEAR})"
echo "$(rm -rf ${DATA_ROOT}/val${YEAR})"
echo "$(rm -rf ${DATA_ROOT}/annotations)"