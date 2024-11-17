from argparse import ArgumentParser
from Tokenizer import Tokenizer
from LargeLanguageModel import LargeLanguageModel
from methods.Prompter import Prompter




if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--prompt', type=str, required=True)
    args = parser.parse_args()
    
    # tokenizer = Tokenizer('mistralai/Mistral-7B-v0.1')
    # llm = LargeLanguageModel('gordicaleksa/SlovenianGPT')
    
    tokenizer = Tokenizer('meta-llama/Meta-Llama-3.1-8B')
    llm = LargeLanguageModel('meta-llama/Meta-Llama-3.1-8B')
    prompter = Prompter(llm, tokenizer)
    print(prompter.prompt(args.prompt))
    print(tokenizer.bos_id)
    print(tokenizer.eos_id)
    print(tokenizer.pad_id)