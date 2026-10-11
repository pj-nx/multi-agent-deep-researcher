# Copyright (c) 2026 C10 Hackathon G1
"""Bag-of-words profanity check (the first version of the guardrail): no dependencies, no model.

The text is lower-cased, common obfuscations (``f*ck``, ``sh1t``) are normalised, and the resulting words
are matched whole against ``BAD_WORDS``. Whole-word matching keeps domain vocabulary ("sex education",
"cocaine use", "Scunthorpe") from tripping the filter.
"""
from __future__ import annotations

import re

BAD_WORDS = frozenset({
    "fuck", "fucks", "fucked", "fucker", "fuckers", "fucking", "motherfucker", "motherfucking",
    "shit", "shits", "shitty", "bullshit", "horseshit", "shithead",
    "bitch", "bitches", "bastard", "bastards", "asshole", "assholes", "arsehole",
    "dick", "dickhead", "cunt", "cunts", "twat", "wanker", "prick",
    "piss", "pissed", "slut", "sluts", "whore", "whores",
    "damn", "goddamn", "crap", "douche", "douchebag", "jackass", "dumbass", "retard", "retarded",
    "nigger", "nigga", "faggot", "fag", "dyke", "spic", "chink", "kike",
})
_LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s",
                       "*": "u", "!": "i"})
_WORD = re.compile(r"[a-z0-9@$*!]+")


def has_profanity(text: str) -> bool:
    for token in _WORD.findall(text.lower()):
        if token in BAD_WORDS or token.translate(_LEET) in BAD_WORDS:
            return True
    return False
