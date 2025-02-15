#!/bin/bash

rm -rf /testing/whisper
cp -r /slollmasr/whisper /testing/whisper
cd /testing/whisper || exit
pip install -e .
cd - || exit