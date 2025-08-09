import string


def sanitize_string(text: str) -> str:
    return text.strip().translate(str.maketrans('', '', string.punctuation)).lower()