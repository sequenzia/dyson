#!/bin/bash

TEMP_DIR=~/.temp
SRC_DIR=~/src

if [ -d "$TEMP_DIR" ]; then
rm -r $TEMP_DIR
fi

if [ -d "$SRC_DIR" ]; then
rm -r $SRC_DIR
fi

mkdir $TEMP_DIR
mkdir $SRC_DIR

git clone https://github.com/sequenzia/photon $TEMP_DIR
git clone https://github.com/sequenzia/photon $TEMP_DIR
