# TOSELF: NEED TO PARSE JSON RIGHT OR THINGS LIKE SLASHES WILL BE BAD!!!

import os
import glob
import json
import re
import ast
from typing import Any, Dict

# --- Settings you might tweak ---
INPUT_DIR = "train_dirty"        # folder containing .jsonl files
OUTPUT_DIR = "train_clean"   # folder to write JSON arrays into

MIN_CHARS_AFTER_CLEAN = 200  # drop super-short docs after cleaning (set 0 to disable)

# Wiki40B markers we want to remove/convert
MARKER_REPLACEMENTS = [
    ("_NEWLINE__NEWLINE_", "\n\n"),
    ("_NEWLINE_", "\n"),
    ("_START_ARTICLE_", "\n"),
    ("_START_SECTION_", "\n"),
    ("_START_PARAGRAPH_", "\n"),
]

# If these English headings appear, drop that heading and everything after it.
# (Wikipedia often ends with these sections, which are not great for LLM pretraining.)
CUT_AT_HEADINGS = {
    "references",
    "external links",
    "see also",
    "further reading",
    "notes",
    "bibliography",
    "sources",
}

heading_line_re = re.compile(r"^\s*([A-Za-z][A-Za-z ]{2,40})\s*$")


def decode_bytes_literal(value: Any) -> Any:
    """
    Some Wiki40B HF outputs (in your files) store fields as strings like:
      'b"\\n_START_ARTICLE_\\n..."' or "b'Q1234'"
    This converts those into proper Unicode strings.
    """
    if not isinstance(value, str):
        return value

    if value.startswith("b'") or value.startswith('b"'):
        try:
            b = ast.literal_eval(value)
            if isinstance(b, (bytes, bytearray)):
                return b.decode("utf-8", errors="replace")
        except Exception:
            # fall through and return original string if it can't be parsed
            return value

    return value


def clean_text(text: str) -> str:
    # 1) Decode bytes-literal if needed
    text = decode_bytes_literal(text)

    if not isinstance(text, str):
        text = str(text)

    # 2) Replace Wiki40B marker tokens
    for old, new in MARKER_REPLACEMENTS:
        text = text.replace(old, new)

    # 3) Remove any leftover ALLCAPS marker-like tokens (conservative)
    # e.g. _START_SOMETHING_ or _SOME_TAG_
    text = re.sub(r"_[A-Z][A-Z_]{2,50}_", "\n", text)

    # 4) Normalize newlines
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 5) Cut off trailing Wikipedia meta sections if present (in English)
    lines = text.split("\n")
    kept_lines = []
    for line in lines:
        m = heading_line_re.match(line)
        if m:
            heading = m.group(1).strip().lower()
            if heading in CUT_AT_HEADINGS:
                break
        kept_lines.append(line)
    text = "\n".join(kept_lines)

    # 6) Clean spacing: collapse repeated spaces/tabs inside lines
    # text = re.sub(r"[ \t]+", " ", text)
    text = text.replace("\t", " ")
    text = re.sub(r"\s+", " ", text)   # collapses ALL whitespace, including newlines

    # 7) Collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def clean_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    # Decode common fields if they are bytes-literal strings
    for k in ("wikidata_id", "version_id"):
        if k in rec:
            rec[k] = decode_bytes_literal(rec[k])

    # Clean text field
    if "text" in rec:
        rec["text"] = clean_text(rec["text"])

    return rec


def convert_file(in_path: str, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    written = 0
    dropped = 0

    with open(in_path, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        fout.write("[\n")
        first = True

        for line in fin:
            line = line.strip()
            if not line:
                continue

            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                # Skip malformed lines
                dropped += 1
                continue

            rec = clean_record(rec)

            if MIN_CHARS_AFTER_CLEAN and len(rec.get("text", "")) < MIN_CHARS_AFTER_CLEAN:
                dropped += 1
                continue

            if not first:
                fout.write(",\n")
            json.dump(rec, fout, ensure_ascii=False)
            first = False
            written += 1

        fout.write("\n]\n")

    print(f"{os.path.basename(in_path)} -> {os.path.basename(out_path)} | kept={written} dropped={dropped}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.jsonl")))

    if not files:
        print(f"No .jsonl files found in: {INPUT_DIR}")
        return

    for in_path in files:
        base = os.path.splitext(os.path.basename(in_path))[0]
        out_path = os.path.join(OUTPUT_DIR, base + ".json")
        convert_file(in_path, out_path)


if __name__ == "__main__":
    main()
