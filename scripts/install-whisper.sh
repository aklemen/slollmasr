#!/bin/bash

cp -r /slollmasr/whisper /testing/whisper
cd /testing/whisper || exit
pip install -e .
cd - || exit