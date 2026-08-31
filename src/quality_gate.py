import re


def clean_answer(text: str) -> str:
    text = text or ""
    text = re.sub(r"<think\b[^>]*>.*?</think\s*>", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<think\b[^>]*>.*$", "", text, flags=re.IGNORECASE | re.DOTALL)
    return re.sub(r"\s+", " ", text).strip()


def repeated(text: str) -> bool:
    words = re.findall(r"\w+", text.lower(), flags=re.UNICODE)
    if len(words) < 28:
        return False
    phrases = [" ".join(words[i:i + 5]) for i in range(len(words) - 4)]
    return len(phrases) - len(set(phrases)) >= 3


def validate_answer(text: str, document_mode: bool) -> tuple[bool, str]:
    cleaned = clean_answer(text)
    if len(cleaned) < 3 or len(cleaned) > 3500:
        return False, "Yanıt üretilemedi. Lütfen soruyu farklı bir biçimde tekrar deneyin."
    if repeated(cleaned):
        return False, cleaned
    return True, cleaned
