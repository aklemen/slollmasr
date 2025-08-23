#!/bin/bash

rm -rf /testing/whisper
cp -r /slollmasr/beam_search/whisper/whisper_package /testing/whisper
cd /testing/whisper || exit
pip install -e .
cd - || exit