from LargeLanguageModel import LargeLanguageModel
from Tokenizer import Tokenizer
from methods.Prompter import Prompter


def test_prompt():
    llm = LargeLanguageModel('openai-community/gpt2')
    tokenizer = Tokenizer('openai-community/gpt2')
    prompter = Prompter(llm=llm, tokenizer=tokenizer)

    input_text = "Hey, do you want to go watch"

    result = prompter.prompt(input_text)

    print('*' * 30)
    print('\n')
    print(result)
    print('\n')
    print('*' * 30)
