import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, required=True)
    parser.add_argument('--llm_name', type=str, required=True)
    parser.add_argument('--tokenizer_name', type=str, required=False)
    parser.add_argument('--manifest_file_path', type=str, required=True)
    parser.add_argument('--beams_file_paths', nargs='+', type=str, required=True)
    parser.add_argument('--beam_sizes', nargs='+', type=int, required=True)
    parser.add_argument('--results_dir_path', type=str, required=True)

    parser.add_argument('--alphas', nargs='+', type=float, required=False)
    parser.add_argument('--betas', nargs='+', type=float, required=False)

    return parser.parse_args()