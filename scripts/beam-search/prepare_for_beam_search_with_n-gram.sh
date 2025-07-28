#!/bin/bash

## Install decoders

apt-get update && apt-get upgrade -y && apt-get install -y swig && rm -rf /var/lib/apt/lists/*

git clone https://github.com/NVIDIA/OpenSeq2Seq && \
cd OpenSeq2Seq && \
git checkout ctc-decoders && \
cd .. && \
mv OpenSeq2Seq/decoders /opt/NeMo/ && \
rm -rf OpenSeq2Seq && \
cd /opt/NeMo/decoders && \
cp /opt/NeMo/scripts/installers/setup_os2s_decoders.py ./setup.py && \
./setup.sh


## Install KenLM

apt-get install -y libeigen3-dev libboost-all-dev
cd /usr/local || exit
git clone https://github.com/kpu/kenlm.git
cd kenlm && mkdir build && cd build && cmake .. && make -j

pip3 install git+https://github.com/kpu/kenlm.git
