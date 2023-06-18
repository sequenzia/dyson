#!/bin/bash

TEMP_DIR=/home/zadmin/.temp
SRC_DIR=/home/zadmin/src

if [ -d $TEMP_DIR ]; then
    chmod 777 -R $TEMP_DIR
    rm -r $TEMP_DIR
fi

if [ -d $SRC_DIR ]; then
    chmod 777 -R $SRC_DIR
    rm -r $SRC_DIR
fi

mkdir $TEMP_DIR
mkdir $SRC_DIR

git clone https://github.com/sequenzia/photon $TEMP_DIR/photon
git clone https://github.com/sequenzia/dyson $TEMP_DIR/dyson

cp -r $TEMP_DIR/photon/photon $SRC_DIR/photon
cp -r $TEMP_DIR/dyson/dyson $SRC_DIR/dyson