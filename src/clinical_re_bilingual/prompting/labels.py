"""Canonical and bilingual surface forms for relation labels."""

from __future__ import annotations

import re

from clinical_re_bilingual.schema import Language, RelationLabel

ENGLISH_NAMES = {
    RelationLabel.TrIP: "TREATMENT IMPROVES MEDICAL PROBLEM",
    RelationLabel.TrWP: "TREATMENT WORSENS MEDICAL PROBLEM",
    RelationLabel.TrCP: "TREATMENT CAUSES MEDICAL PROBLEM",
    RelationLabel.TrAP: "TREATMENT IS ADMINISTERED FOR MEDICAL PROBLEM",
    RelationLabel.TrNAP: "TREATMENT IS NOT ADMINISTERED BECAUSE OF MEDICAL PROBLEM",
    RelationLabel.PIP: "MEDICAL PROBLEM INDICATES MEDICAL PROBLEM",
    RelationLabel.TeRP: "TEST REVEALS MEDICAL PROBLEM",
    RelationLabel.TeCP: "TEST CONDUCTED TO INVESTIGATE MEDICAL PROBLEM",
}

TURKISH_CODES = {
    RelationLabel.TrIP: "TdvİP",
    RelationLabel.TrWP: "TdvKP",
    RelationLabel.TrCP: "TdvNOP",
    RelationLabel.TrAP: "PİUTdv",
    RelationLabel.TrNAP: "PNUMTdv",
    RelationLabel.PIP: "PGP",
    RelationLabel.TeRP: "TstOÇP",
    RelationLabel.TeCP: "PİUTst",
}

TURKISH_NAMES = {
    RelationLabel.TrIP: "TEDAVİ İYİLEŞTİRİR MEDİKAL PROBLEMİ",
    RelationLabel.TrWP: "TEDAVİ KÖTÜLEŞTİRİR MEDİKAL PROBLEMİ",
    RelationLabel.TrCP: "TEDAVİ NEDEN OLUR MEDİKAL PROBLEME",
    RelationLabel.TrAP: "MEDİKAL PROBLEM İÇİN UYGULANDI TEDAVİ",
    RelationLabel.TrNAP: "MEDİKAL PROBLEM NEDENİYLE UYGULANMADI TEDAVİ",
    RelationLabel.PIP: "MEDİKAL PROBLEM GÖSTERİR DİĞER MEDİKAL PROBLEMİ",
    RelationLabel.TeRP: "TEST ORTAYA ÇIKARIR MEDİKAL PROBLEMİ",
    RelationLabel.TeCP: "MEDİKAL PROBLEM İÇİN UYGULANDI TEST",
}


def relation_code(label: RelationLabel, language: Language) -> str:
    return label.value if language is Language.ENGLISH else TURKISH_CODES[label]


def relation_name(label: RelationLabel, language: Language) -> str:
    return ENGLISH_NAMES[label] if language is Language.ENGLISH else TURKISH_NAMES[label]


def _normalise(value: str) -> str:
    return re.sub(r"[^\wİıŞşĞğÜüÖöÇç]+", " ", value.casefold()).strip()


def parse_relation_label(response: str) -> RelationLabel:
    """Parse one canonical label from a model response or raise on ambiguity."""

    normalised = _normalise(response)
    matches: set[RelationLabel] = set()
    for label in RelationLabel:
        aliases = {
            label.value,
            ENGLISH_NAMES[label],
            TURKISH_CODES[label],
            TURKISH_NAMES[label],
        }
        for alias in aliases:
            pattern = rf"(?:^|\s){re.escape(_normalise(alias))}(?:$|\s)"
            if re.search(pattern, normalised):
                matches.add(label)
    if len(matches) != 1:
        found = sorted(label.value for label in matches)
        raise ValueError(f"expected exactly one relation label; found {found}")
    return matches.pop()
