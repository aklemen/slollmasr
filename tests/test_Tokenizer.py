from Tokenizer import Tokenizer


def test_tokenizer_initialization():
    model_name = 'openai-community/gpt2'
    tokenizer = Tokenizer(name=model_name)

    assert tokenizer.pad_id == tokenizer.tokenizer.pad_token_id
    assert tokenizer.bos_id == tokenizer.tokenizer.bos_token_id
    assert tokenizer.eos_id == tokenizer.tokenizer.eos_token_id
