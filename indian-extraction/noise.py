"""
noise.py - makes clean synthetic text look like messy real-world text
(typos, OCR mistakes, odd line breaks).

IMPORTANT: we never touch the words/numbers that are part of the answer
("protected" tokens), so the correct JSON is always still findable in the text.
"""
import random

OCR_CONFUSIONS = {"o": "0", "l": "1", "i": "1", "s": "5", "b": "6"}
NOISE_P = {0: 0.0, 1: 0.05, 2: 0.15}   # chance a normal word gets damaged


def _strip(tok: str) -> str:
    return tok.strip(".,:;()-|").lower()


def protected_words(rendered_values):
    words = set()
    for value in rendered_values:
        for w in str(value).split():
            words.add(_strip(w))
    return words


def _damage_word(word: str, rng: random.Random) -> str:
    op = rng.choice(["swap", "drop", "ocr", "upper"])
    if op == "swap" and len(word) >= 4:
        i = rng.randrange(len(word) - 1)
        return word[:i] + word[i + 1] + word[i] + word[i + 2:]
    if op == "drop" and len(word) >= 4:
        i = rng.randrange(len(word))
        return word[:i] + word[i + 1:]
    if op == "ocr":
        spots = [i for i, c in enumerate(word.lower()) if c in OCR_CONFUSIONS]
        if spots:
            i = rng.choice(spots)
            return word[:i] + OCR_CONFUSIONS[word[i].lower()] + word[i + 1:]
    if op == "upper":
        return word.upper()
    return word


def add_noise(text: str, level: int, rendered_values, rng: random.Random) -> str:
    p = NOISE_P[level]
    if p == 0:
        return text
    prot = protected_words(rendered_values)
    out_lines = []
    for line in text.split("\n"):
        tokens = line.split(" ")
        new_tokens, is_prot = [], []
        for tok in tokens:
            safe = (any(c.isdigit() for c in tok) or _strip(tok) in prot
                    or not tok.isalpha())
            if not safe and rng.random() < p:
                tok = _damage_word(tok, rng)
            new_tokens.append(tok)
            is_prot.append(safe)
        # rebuild line; sometimes break the line or glue words together
        line_out = new_tokens[0] if new_tokens else ""
        for i in range(1, len(new_tokens)):
            sep = " "
            if not (is_prot[i] and is_prot[i - 1]) and rng.random() < p:
                sep = rng.choice(["\n", ""])
            line_out += sep + new_tokens[i]
        out_lines.append(line_out)
    return "\n".join(out_lines)
