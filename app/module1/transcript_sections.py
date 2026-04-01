from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class ParsedTranscriptSections:
    findings: str
    impression: str


SECTION_PATTERN = re.compile(
    r"(?P<label>\b(?:bulgular|bulgu|sonu(?:ç|c)|izlenim|impression)\b)\s*[:\-]?",
    flags=re.IGNORECASE,
)


def parse_transcript_sections(transcript: str) -> ParsedTranscriptSections:
    cleaned = " ".join(transcript.strip().split())
    if not cleaned:
        return ParsedTranscriptSections(findings="", impression="")

    matches = list(SECTION_PATTERN.finditer(cleaned))
    if not matches:
        return ParsedTranscriptSections(findings=cleaned, impression="")

    findings_parts: list[str] = []
    impression_parts: list[str] = []

    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(cleaned)
        content = cleaned[start:end].strip(" :-\n\t")
        if not content:
            continue

        label = match.group("label").lower()
        if re.fullmatch(r"sonu(?:ç|c)|izlenim|impression", label, flags=re.IGNORECASE):
            impression_parts.append(content)
        else:
            findings_parts.append(content)

    findings = " ".join(findings_parts).strip()
    impression = " ".join(impression_parts).strip()

    if not findings and impression:
        findings = cleaned

    return ParsedTranscriptSections(findings=findings, impression=impression)
