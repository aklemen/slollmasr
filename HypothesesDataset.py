import torch
import pandas as pd
import numpy as np
import json
import logging


class HypothesesDataset(torch.utils.data.Dataset):
    def __init__(self, hypotheses_file_path, manifest_file_path, tokenizer, beam_size, max_seq_length):
        self.hypotheses = pd.read_csv(hypotheses_file_path, delimiter="\t", header=None)
        self.ground_truths = self._read_manifest(manifest_file_path)

        self.tokenizer = tokenizer
        self.beam_size = beam_size
        self.max_seq_length = max_seq_length

        self.pad_id = self._get_pad_id(tokenizer)

    def __len__(self):
        return len(self.hypotheses)

    def __getitem__(self, idx):
        ground_truth = self.ground_truths[idx // self.beam_size]
        hypothesis = str(self.hypotheses[0][idx])
        asr_score = self.hypotheses[1][idx]
        hypothesis_ids = self._convert_text_to_ids(hypothesis)
        input_ids = self._get_input_ids(hypothesis_ids)
        input_mask = self._get_input_mask(hypothesis_ids)
        return ground_truth, hypothesis, asr_score, input_ids, input_mask

    def get_beam_size(self):
        return self.beam_size

    def _convert_text_to_ids(self, hypothesis_text):
        tokens = self.tokenizer.tokenize(hypothesis_text)
        hypothesis_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        if hasattr(self.tokenizer, "bos_id") and self.tokenizer.bos_id is not None:
            hypothesis_ids = [self.tokenizer.bos_id] + hypothesis_ids
        if hasattr(self.tokenizer, "eos_id") and self.tokenizer.eos_id is not None:
            hypothesis_ids = hypothesis_ids + [self.tokenizer.eos_id]
        return hypothesis_ids

    def _read_manifest(self, manifest_file_path):
        ground_truths = []
        with open(manifest_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                manifest_entry = json.loads(line)
                ground_truth = manifest_entry['text']
                ground_truths.append(ground_truth)
        return ground_truths

    def _get_pad_id(self, tokenizer):
        if hasattr(tokenizer, "pad_id") and tokenizer.pad_id is not None:
            return tokenizer.pad_id
        elif hasattr(tokenizer, "eos_id") and tokenizer.eos_id is not None:
            return tokenizer.eos_id
        else:
            logging.info(f"Using 0 as pad_id as the tokenizer has no pad_id or eos_id.")
            return 0

    def _get_input_ids(self, hypothesis_ids):
        input_ids = [self.pad_id] * self.max_seq_length
        input_ids[: len(hypothesis_ids)] = hypothesis_ids
        input_ids = np.array(input_ids)
        return input_ids

    def _get_input_mask(self, hypothesis_ids):
        input_mask = np.zeros(self.max_seq_length)
        input_mask[: len(hypothesis_ids)] = 1
        return input_mask
