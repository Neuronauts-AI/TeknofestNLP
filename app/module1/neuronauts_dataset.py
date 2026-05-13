from __future__ import annotations

import re
from pathlib import Path


DEFAULT_NEURONAUTS_ROOT = Path("Neuronauts")
REPORTS_DIR_NAME = "Raporlar"
IMAGES_DIR_NAME = "Görüntüler"
AUDIO_DIR_NAME = "Sesler"


RTF_IGNORABLE_DESTINATIONS = {
    "fonttbl",
    "colortbl",
    "datastore",
    "themedata",
    "stylesheet",
    "info",
    "pict",
    "object",
    "nonshppict",
    "shp",
    "shpinst",
    "background",
    "userprops",
}


def normalize_case_id(path: Path) -> str:
    match = re.search(r"hasta[\s-]*(\d+)", path.stem, flags=re.IGNORECASE)
    if match:
        return match.group(1)
    return path.stem


def rtf_to_text(rtf_text: str) -> str:
    output: list[str] = []
    stack: list[tuple[int, bool]] = []
    ignorable = False
    unicode_skip = 1
    current_skip = 0
    index = 0

    while index < len(rtf_text):
        char = rtf_text[index]
        if char == "{":
            stack.append((unicode_skip, ignorable))
            index += 1
            continue
        if char == "}":
            if stack:
                unicode_skip, ignorable = stack.pop()
            index += 1
            continue
        if char != "\\":
            if current_skip > 0:
                current_skip -= 1
            elif not ignorable:
                output.append(char)
            index += 1
            continue

        index += 1
        if index >= len(rtf_text):
            break

        command_char = rtf_text[index]
        if command_char in "\\{}":
            if not ignorable:
                output.append(command_char)
            index += 1
            continue
        if command_char == "'" and index + 2 < len(rtf_text):
            hex_value = rtf_text[index + 1 : index + 3]
            if not ignorable and current_skip <= 0:
                try:
                    output.append(bytes.fromhex(hex_value).decode("cp1254", errors="ignore"))
                except ValueError:
                    pass
            index += 3
            continue
        if command_char == "*":
            ignorable = True
            index += 1
            continue
        if command_char in "\n\r":
            index += 1
            continue

        match = re.match(r"([a-zA-Z]+)(-?\d+)? ?", rtf_text[index:])
        if not match:
            index += 1
            continue

        word = match.group(1)
        argument = match.group(2)
        index += len(match.group(0))

        if word in RTF_IGNORABLE_DESTINATIONS:
            ignorable = True
        elif word == "uc" and argument is not None:
            unicode_skip = int(argument)
        elif word == "u" and argument is not None:
            value = int(argument)
            if value < 0:
                value += 65536
            if not ignorable:
                output.append(chr(value))
            current_skip = unicode_skip
        elif word in {"par", "line"}:
            if not ignorable:
                output.append("\n")
        elif word == "tab":
            if not ignorable:
                output.append("\t")

    plain_text = "".join(output)
    plain_text = re.sub(r"[ \t]+", " ", plain_text)
    plain_text = re.sub(r"\n\s*\n+", "\n\n", plain_text)
    return plain_text.strip()


def extract_report_body(full_text: str) -> str:
    text = full_text.replace("\r\n", "\n").replace("\r", "\n")
    start_match = re.search(r"Rapor Bilgileri", text, flags=re.IGNORECASE)
    end_match = re.search(r"Doktor Bilgileri", text, flags=re.IGNORECASE)

    if start_match:
        text = text[start_match.end() :]
        end_match = re.search(r"Doktor Bilgileri", text, flags=re.IGNORECASE)
    if end_match:
        text = text[: end_match.start()]

    lines = [line.strip() for line in text.splitlines()]
    cleaned_lines = [line for line in lines if line and line not in {":", ";"}]
    return "\n".join(cleaned_lines).strip()


def read_rtf_report(path: Path) -> str:
    raw_text = path.read_text(encoding="cp1254", errors="ignore")
    return extract_report_body(rtf_to_text(raw_text))


def _case_file_map(directory: Path, pattern: str) -> dict[str, Path]:
    if not directory.exists():
        return {}
    return {normalize_case_id(path): path for path in directory.glob(pattern)}


def load_neuronauts_cases(root: Path = DEFAULT_NEURONAUTS_ROOT) -> list[dict[str, str]]:
    reports_dir = root / REPORTS_DIR_NAME
    images_by_case = _case_file_map(root / IMAGES_DIR_NAME, "*.png")
    audio_by_case = _case_file_map(root / AUDIO_DIR_NAME, "*.wav")

    if not reports_dir.exists():
        return []

    cases: list[dict[str, str]] = []
    for report_path in sorted(reports_dir.glob("*.rtf"), key=lambda path: int(normalize_case_id(path))):
        case_id = normalize_case_id(report_path)
        report_text = read_rtf_report(report_path)
        if not report_text:
            continue
        image_path = images_by_case.get(case_id)
        audio_path = audio_by_case.get(case_id)
        cases.append(
            {
                "id": case_id,
                "findings_tr": report_text,
                "impression_tr": "",
                "report_tr": f"Bulgular:\n{report_text}",
                "source_report_path": str(report_path),
                "image_path": str(image_path) if image_path else "",
                "audio_path": str(audio_path) if audio_path else "",
            }
        )
    return cases
