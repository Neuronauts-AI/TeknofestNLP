import re
from dataclasses import dataclass
from pathlib import Path


POSITIVE_PATTERNS = [
    r"\bizlenmiştir\b",
    r"\bsaptanmamıştır\b",
    r"\bnormal sınırlardadır\b",
    r"\bdoğaldır\b",
    r"\buyumludur\b",
    r"\bmevcuttur\b",
    r"\bizlenmektedir\b",
    r"\bkonturlar\b",
    r"\bsinüs\b",
    r"\bparankimi\b",
]

NEGATIVE_PATTERNS = [
    r"\bdüşündürür\b",
    r"\bbence\b",
    r"\bsanki\b",
    r"\bgaliba\b",
    r"\byani\b",
    r"\bçok kötü\b",
    r"\bharika\b",
    r"\benteresan\b",
]

SECTION_PATTERNS = [
    r"\bsonuç\b",
    r"\bbulgular\b",
]

REFERENCE_STYLE_PATH = Path("docs/reference_report_style.txt")


@dataclass
class StyleClassification:
    label: str
    score: float
    reasons: list[str]


def normalize_text(text: str) -> str:
    return " ".join((text or "").strip().split()).lower()


def load_reference_style() -> str:
    if not REFERENCE_STYLE_PATH.exists():
        return ""
    return REFERENCE_STYLE_PATH.read_text(encoding="utf-8").strip()


def classify_report_style(findings: str, impression: str) -> StyleClassification:
    findings = (findings or "").strip()
    impression = (impression or "").strip()
    combined = f"{findings}\n{impression}".strip()
    normalized = normalize_text(combined)
    reference_text = load_reference_style()
    reference_normalized = normalize_text(reference_text)

    score = 0.0
    reasons: list[str] = []

    positive_hits = sum(
        1 for pattern in POSITIVE_PATTERNS if re.search(pattern, normalized, flags=re.IGNORECASE)
    )
    negative_hits = sum(
        1 for pattern in NEGATIVE_PATTERNS if re.search(pattern, normalized, flags=re.IGNORECASE)
    )
    section_hits = sum(
        1 for pattern in SECTION_PATTERNS if re.search(pattern, normalized, flags=re.IGNORECASE)
    )
    reference_positive_hits = sum(
        1 for pattern in POSITIVE_PATTERNS if re.search(pattern, reference_normalized, flags=re.IGNORECASE)
    )
    matched_reference_patterns = min(positive_hits, reference_positive_hits)

    if findings:
        score += 0.2
        reasons.append("Bulgular metni mevcut.")
    if impression:
        score += 0.2
        reasons.append("Sonuç/izlenim metni mevcut.")

    if positive_hits:
        score += min(0.4, positive_hits * 0.08)
        reasons.append(f"Radyoloji raporu diline uygun {positive_hits} ifade bulundu.")

    if reference_text and matched_reference_patterns:
        score += min(0.15, matched_reference_patterns * 0.03)
        reasons.append("Referans rapor üslubuyla ortak kalıplar bulundu.")
    elif reference_text:
        reasons.append("Referans rapor metni yüklendi ancak ortak kalıp az bulundu.")

    if section_hits:
        score += min(0.1, section_hits * 0.05)
        reasons.append("Rapor yapısına uygun bölüm dili kullanılmış.")

    if negative_hits:
        score -= min(0.35, negative_hits * 0.12)
        reasons.append(f"Doğal klinik rapor diline uymayan {negative_hits} ifade bulundu.")

    if len(combined) < 40:
        score -= 0.15
        reasons.append("Metin çok kısa; gerçek rapor hissi zayıf olabilir.")

    label = "uygun" if score >= 0.45 else "uygun_degil"
    score = max(0.0, min(1.0, round(score, 4)))
    return StyleClassification(label=label, score=score, reasons=reasons)
