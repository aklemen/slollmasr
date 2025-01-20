#!/bin/bash

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

curl -o /opt/NeMo/scripts/asr_language_modeling/ngram_lm/eval_beamsearch_ngram_ctc.py https://raw.githubusercontent.com/aklemen/NeMo/decoding-fix/scripts/asr_language_modeling/ngram_lm/eval_beamsearch_ngram_ctc.py
curl -o /opt/NeMo/nemo/collections/asr/parts/submodules/ctc_beam_decoding.py https://raw.githubusercontent.com/aklemen/NeMo/decoding-fix/nemo/collections/asr/parts/submodules/ctc_beam_decoding.py

sed -i 's/asr_model._wer/asr_model.wer/g' /opt/NeMo/scripts/asr_language_modeling/ngram_lm/eval_beamsearch_ngram_ctc.py