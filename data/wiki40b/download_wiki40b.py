from datasets import load_dataset
import json
import os

LANG_MB = {
    # latin
    "en": 10,
    "fr": 5,
    "es": 5,
    "de": 5,
    "it": 5,
    "pt": 0,
    "nl": 0,
    "pl": 0,
    # cyrillic
    "ru": 5,
    "uk": 5,
    "bg": 5,
    # arabic
    "ar": 5,
    "fa": 5,
    # han
    "zh-cn": 5,
    "zh-tw": 5,
    # japanese
    "ja": 5,
    # hangul
    "ko": 5,
    # greek
    "el": 0,
    # hebrew
    "he": 0,
    # thai
    "th": 0,
}

SPLIT = "train"
SEED = 42
SHUFFLE_BUFFER = 10_000

def ok(ex) -> bool:
    t = ex.get("text") or ""
    return len(t) > 3000

def bytes_to_mb(b: int) -> float:
    return b / (1024 * 1024)

for lang, target_mb in LANG_MB.items():
    if target_mb > 0:
        out = f"{lang}_{SPLIT}_{target_mb}MB.jsonl"
        target_bytes = int(target_mb * 1024 * 1024)

        ds = load_dataset("wiki40b", lang, split=SPLIT, streaming=True)
        ds = ds.shuffle(seed=SEED, buffer_size=SHUFFLE_BUFFER)

        written = 0
        size_bytes = 0

        with open(out, "w", encoding="utf-8", newline="\n") as f:
            for ex in ds:
                if not ok(ex):
                    continue

                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
                written += 1

                # flush + size check each time so we stop right after passing the threshold
                f.flush()
                size_bytes = os.path.getsize(out)
                if size_bytes >= target_bytes:
                    break

        print(
            f"{lang}: wrote {written} examples -> {out} "
            f"({bytes_to_mb(size_bytes):.2f} MB, target {target_mb} MB)"
        )
