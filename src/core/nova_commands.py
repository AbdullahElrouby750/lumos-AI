import re
import difflib

INTENT_NONE = "INTENT_NONE"
INTENT_ENROLL = "INTENT_ENROLL"
INTENT_FORGET = "INTENT_FORGET"
INTENT_QUIT = "INTENT_QUIT"
INTENT_CANCEL = "INTENT_CANCEL"
INTENT_CONFIRM = "INTENT_CONFIRM"
INTENT_DENY = "INTENT_DENY"
INTENT_WAKE_ONLY = "INTENT_WAKE_ONLY"
INTENT_SCENE = "INTENT_SCENE"
INTENT_TEXT = "INTENT_TEXT"

ENROLL_KEYWORDS = ["enroll", "add", "remember", "register", "save"]
FORGET_KEYWORDS = ["delete", "remove", "forget", "clear"]
QUIT_KEYWORDS = ["quit", "exit", "close", "shutdown", "shut-down", "shut down"]
CANCEL_KEYWORDS = ["cancel", "abort", "stop enrollment", "never mind", "stop"]
CONFIRM_KEYWORDS = ["yes", "yeah", "yep", "sure", "do it", "confirm"]
DENY_KEYWORDS = ["no", "nope", "don't"]
IGNORE_WORDS = {
    "hey", "lumo", "lumos", "please", "the", "a", "an", "to", "of", "me", "my", "can", "you"
}

def _clean_name(name_text):
    name_text = name_text.strip()
    name_text = re.sub(r"[^\w\s'-]", "", name_text)
    tokens = [token for token in name_text.split() if token.lower() not in IGNORE_WORDS]
    if not tokens:
        return None
    return " ".join(tokens).title()


def _extract_name(raw_text, keyword):
    pattern = rf"\b{re.escape(keyword)}\b\s*(?P<name>[\w\s'-]+)"
    match = re.search(pattern, raw_text, re.IGNORECASE)
    if match:
        return _clean_name(match.group("name"))

    # Fallback: use the last non-keyword words after removing command words
    lower_text = raw_text.lower()
    for word in ENROLL_KEYWORDS + FORGET_KEYWORDS + QUIT_KEYWORDS:
        lower_text = lower_text.replace(word, " ")
    tokens = [token for token in re.split(r"\s+", lower_text) if token and token.lower() not in IGNORE_WORDS]
    if not tokens:
        return None
    return _clean_name(" ".join(tokens))


def parse_intent(raw_text):
    """Parse raw voice text into intent and target name."""
    if not raw_text or not raw_text.strip():
        return {"intent": INTENT_NONE, "target_name": None}

    text = raw_text.strip()
    lower_text = text.lower()

    for keyword in QUIT_KEYWORDS:
        if keyword in lower_text:
            return {"intent": INTENT_QUIT, "target_name": None}

    for keyword in CONFIRM_KEYWORDS:
        if keyword in lower_text:
            return {"intent": INTENT_CONFIRM, "target_name": None}

    for keyword in DENY_KEYWORDS:
        if keyword in lower_text:
            return {"intent": INTENT_DENY, "target_name": None}

    for keyword in CANCEL_KEYWORDS:
        if keyword in lower_text:
            return {"intent": INTENT_CANCEL, "target_name": None}

    for keyword in FORGET_KEYWORDS:
        if keyword in lower_text:
            target_name = _extract_name(text, keyword)
            return {"intent": INTENT_FORGET, "target_name": target_name}

    for keyword in ENROLL_KEYWORDS:
        if keyword in lower_text:
            target_name = _extract_name(text, keyword)
            return {"intent": INTENT_ENROLL, "target_name": target_name}

    return {"intent": INTENT_WAKE_ONLY, "target_name": None}
