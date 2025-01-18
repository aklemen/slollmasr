def are_chat_templates_supported(tokenizer):
    return hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None