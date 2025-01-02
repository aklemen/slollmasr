from transformers import AutoTokenizer


class Tokenizer:
    def __init__(self, name: str):
        self.name = name
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=name,
            use_fast=True,
        )

        # Cache special token ids
        self.pad_id = self._get_pad_id()
        self.bos_id = self._get_bos_id()
        self.eos_id = self._get_eos_id()

        # Update the hugging-face tokenizer with custom tokens
        self.tokenizer.pad_token_id = self.pad_id
        self.tokenizer.bos_token_id = self.bos_id
        self.tokenizer.eos_token_id = self.eos_id

    def text_to_ids(self, text):
        tokens = self.tokenizer.tokenize(text)
        ids = self._tokens_to_ids(tokens)
        if self.bos_id is not None:
            ids = [self.bos_id] + ids
        if self.eos_id is not None:
            ids = ids + [self.eos_id]
        return ids

    def are_chat_templates_supported(self):
        return hasattr(self.tokenizer, "chat_template") and self.tokenizer.chat_template is not None

    def set_padding_side(self, side: str):
        self.tokenizer.padding_side = side

    def _get_pad_id(self):
        if hasattr(self.tokenizer, "pad_token") and self.tokenizer.pad_token is not None:
            return self._tokens_to_ids([self.tokenizer.pad_token])[0]
        elif hasattr(self.tokenizer, "eos_token") and self.tokenizer.eos_token is not None:
            print(f"Using eos_id as pad_id as the tokenizer has no pad_token.")
            return self._tokens_to_ids([self.tokenizer.eos_token])[0]
        else:
            print(f"Using 0 as pad_id as the tokenizer has no pad_token or eos_token.")
            return 0

    def _get_bos_id(self):
        if getattr(self.tokenizer, 'bos_token') is None:
            return None
        return self._tokens_to_ids([self.tokenizer.bos_token])[0]

    def _get_eos_id(self):
        if getattr(self.tokenizer, 'eos_token') is None:
            return None
        return self._tokens_to_ids([self.tokenizer.eos_token])[0]

    def _tokens_to_ids(self, tokens: list[str]):
        return self.tokenizer.convert_tokens_to_ids(tokens)
