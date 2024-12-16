from argparse import ArgumentParser
from pathlib import Path

from Tokenizer import Tokenizer
from LargeLanguageModel import LargeLanguageModel
from methods.Prompter import Prompter




if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--llm_name', type=str, required=True)
    parser.add_argument('--tokenizer_name', type=str, required=False)
    parser.add_argument('--manifest_file_path', type=str, required=True)
    parser.add_argument('--beams_file_paths', nargs='+', type=str, required=True)
    parser.add_argument('--beam_sizes', nargs='+', type=int, required=True)
    parser.add_argument('--results_dir_path', type=str, required=True)
    args = parser.parse_args()

    if len(args.beam_sizes) != len(args.beams_file_paths) or len(args.beam_sizes) != len(args.alphas) or len(args.beam_sizes) != len(args.betas):
        raise ValueError("The number of beam_sizes, alphas and betas should be the same as the number of beams_file_paths")

    if args.tokenizer_name is None:
        print(f"Tokenizer name was not given, using LLM name '{args.llm_name}'")
        args.tokenizer_name = args.llm_name

    Path(args.results_dir_path).mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer(args.tokenizer_name)
    llm = LargeLanguageModel(args.llm_name)
    rescorer = PromptSelector(llm)
    calc = MetricsCalculator()

    prompter = Prompter(llm, tokenizer)
    print(prompter.prompt(args.prompt))
    print(tokenizer.bos_id)
    print(tokenizer.eos_id)
    print(tokenizer.pad_id)