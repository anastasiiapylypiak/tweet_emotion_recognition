# src/translation.py
# Language detection + optional German/French → English translation

from typing import List, Tuple

import torch
from langdetect import detect, DetectorFactory
from transformers import MarianMTModel, MarianTokenizer

# Make langdetect deterministic
DetectorFactory.seed = 0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# LANGUAGE DETECTION
# ----------------------------
def detect_language(text: str) -> str:
    """
    Detect language of a single text.
    Returns ISO code like 'en', 'de', 'fr', or 'unknown'.
    """
    if not isinstance(text, str) or not text.strip():
        return "unknown"
    try:
        return detect(text)
    except Exception:
        return "unknown"


def detect_languages(texts: List[str]) -> List[str]:
    """
    Detect language for a list of texts.
    """
    return [detect_language(t) for t in texts]


# ----------------------------
# TRANSLATION
# ----------------------------
_TRANSLATOR_MAP = {
    "de": "Helsinki-NLP/opus-mt-de-en",
    "fr": "Helsinki-NLP/opus-mt-fr-en",
}

_TRANSLATORS = {}  # cache loaded MarianMT models/tokenizers


def load_translator(src_lang: str) -> Tuple[MarianTokenizer, MarianMTModel]:
    """
    Load MarianMT model for given source language ('de' or 'fr').
    """
    if src_lang not in _TRANSLATOR_MAP:
        raise ValueError(f"No translator available for language '{src_lang}'")

    if src_lang in _TRANSLATORS:
        return _TRANSLATORS[src_lang]

    model_name = _TRANSLATOR_MAP[src_lang]
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    model.to(DEVICE)
    model.eval()
    _TRANSLATORS[src_lang] = (tokenizer, model)
    return tokenizer, model


@torch.no_grad()
def translate_to_en(
    texts: List[str],
    src_lang: str,
    batch_size: int = 8,
    max_length: int = 256,
) -> List[str]:
    """
    Translate a list of texts in `src_lang` to English.
    """
    tokenizer, model = load_translator(src_lang)
    translations = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]

        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(DEVICE)

        generated = model.generate(
            **enc,
            max_length=max_length,
            num_beams=4,
            early_stopping=True,
        )

        out = tokenizer.batch_decode(generated, skip_special_tokens=True)
        translations.extend(out)

    return translations


# ----------------------------
# HIGH-LEVEL HELPER
# ----------------------------
def ensure_english_texts(
    texts: List[str],
    translate_german: bool = True,
    translate_french: bool = True,
) -> Tuple[List[str], List[str], List[bool]]:
    """
    Ensure all texts are English.

    Returns:
      - processed_texts: texts in English (translated if needed)
      - languages: detected languages
      - was_translated: bool list, True if translation was applied
    """
    languages = detect_languages(texts)
    processed = list(texts)
    was_translated = [False] * len(texts)

    # German translation
    if translate_german:
        de_indices = [i for i, lang in enumerate(languages) if lang == "de"]
        if de_indices:
            de_texts = [texts[i] for i in de_indices]
            de_translations = translate_to_en(de_texts, "de")
            for idx, translated in zip(de_indices, de_translations):
                processed[idx] = translated
                was_translated[idx] = True

    # French translation
    if translate_french:
        fr_indices = [i for i, lang in enumerate(languages) if lang == "fr"]
        if fr_indices:
            fr_texts = [texts[i] for i in fr_indices]
            fr_translations = translate_to_en(fr_texts, "fr")
            for idx, translated in zip(fr_indices, fr_translations):
                processed[idx] = translated
                was_translated[idx] = True

    return processed, languages, was_translated