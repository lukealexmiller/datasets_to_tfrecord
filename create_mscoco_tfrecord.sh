#!/bin/bash

# Specify data root and Syno mount point
DATA_ROOT="/root/data/mscoco"
SYNO_ZIP="${DATA_ROOT}/zips"

# Specify MSCOCO dataset year
YEAR="2014"

# Get data from MSCOCO
if [ ! -d "${SYNO_ZIP}" ]; then 
  echo "$(mkdir -p ${SYNO_ZIP})" ;
fi  && \

URL_TRAIN="http://images.cocodataset.org/zips/train${YEAR}.zip"
URL_VAL="http://images.cocodataset.org/zips/train${YEAR}.zip"
URL_ANNO="http://images.cocodataset.org/zips/train${YEAR}.zip"

wget "${URL_TRAIN}" -O "${SYNO_ZIP}" && \
wget "${URL_VAL}" -O "${SYNO_ZIP}" && \
wget "${URL_ANNO}" -O "${SYNO_ZIP}" && \

# Construct arguments to pass to python
ARGS="--data_root=${DATA_ROOT} --output_root=${DATA_ROOT} --year=${YEAR} "
ARGS+="--shuffle=True --num_val=8000 --train_shards=64 --val_shards=4 --test_shards=0 --num_threads=8"

# Install MSCOCO API
if [ ! -d "/root/data/scripts/cocoapi" ]; then 
  echo "$(chmod +x install_mscoco_api.sh)" && \
  ./install_mscoco_api.sh ; 
fi  && \

# Unzip data located on Syno to local directories
echo "$(unzip ${SYNO_ZIP}/val${YEAR}.zip -d ${DATA_ROOT})" && \
echo "$(unzip ${SYNO_ZIP}/train${YEAR}.zip -d ${DATA_ROOT})" && \
echo "$(unzip ${SYNO_ZIP}/annotations_trainval${YEAR}.zip -d ${DATA_ROOT})" && \

# Create TF Records
echo "$(python ./create_mscoco_tf_record.py ${ARGS})"

# Remove temporary local directories
echo "$(rm -rf ${DATA_ROOT}/train${YEAR})"
echo "$(rm -rf ${DATA_ROOT}/val${YEAR})"
echo "$(rm -rf ${DATA_ROOT}/annotations)"