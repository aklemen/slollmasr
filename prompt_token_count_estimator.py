import json
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--tokenizer_name', type=str, required=True, help='Name of the HF tokenizer')
    parser.add_argument('--manifest_file_paths', nargs='+', type=str, required=True, help='Paths to the manifest files')
    parser.add_argument('--token_count_multiplier', type=int, default=1, help='Multiplier for the token count')
    parser.add_argument('--token_count_to_add', type=int, default=0, help='Token count to add to each text')
    parser.add_argument('--num_bins', type=int, default=50, help='Token count to add to each text')
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True)

    fig, ax = plt.subplots()
    token_counts = []
    for manifest_file_path in args.manifest_file_paths:
        print(f"Processing file '{manifest_file_path}' ...")
        with open(manifest_file_path, 'r') as file:
            for line in file:
                data = json.loads(line)
                text = data['text']
                tokens = tokenizer.tokenize(text)
                tokens_count = len(tokens) * args.token_count_multiplier + args.token_count_to_add
                token_counts.append(tokens_count)

        ax.clear()
        ax.hist(token_counts, bins=args.num_bins, edgecolor='black')
        ax.set_title('Token Count Distribution')
        ax.set_xlabel('Number of Tokens')
        ax.set_ylabel('Frequency')
        plt.draw()
        plt.pause(0.1)


    print(f"Total number of texts: {len(token_counts)}")
    print(f"Minimum token count: {min(token_counts)}")
    print(f"Maximum token count: {max(token_counts)}")
    print(f"Average token count: {sum(token_counts) / len(token_counts)}")
    print(f"Sum of all token counts: {sum(token_counts)}")

    step = max(token_counts) // args.num_bins
    bins = np.arange(0, max(token_counts) + step, step)
    hist, bin_edges = np.histogram(token_counts, bins=bins)

    # Print the number of texts in each bin
    for i in range(len(hist)):
        print(f"Number of texts with token count {bin_edges[i]}-{bin_edges[i+1]}: {hist[i]}")

    plt.show()