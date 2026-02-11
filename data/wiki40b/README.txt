README
======

Overview
--------
This repo contains two Python scripts for downloading and cleaning Wiki40B data:

1) download_wiki40b.py
   - Streams the Hugging Face "wiki40b" dataset for a set of languages.
   - Shuffles the stream, filters to longer documents, and writes examples to JSONL files
     until each file reaches a target size (in MB).

2) clean_wiki40b.py
   - Reads .jsonl files from a folder, cleans Wiki40B marker tokens / common trailing
     Wikipedia sections, and writes the result as a JSON array (.json) per input file.

Dependencies
------------
- Python 3.9+
- Third-party packages:
  - datasets

Environment setup
-----------------

macOS / Linux
~~~~~~~~~~~~~

    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip
    pip install -r requirements.txt

Windows (PowerShell)
~~~~~~~~~~~~~~~~~~~~

    py -m venv .venv
    .\.venv\Scripts\Activate.ps1
    py -m pip install --upgrade pip
    pip install -r requirements.txt

Script details
--------------

1) download_wiki40b
~~~~~~~~~~~~~~~~~~~
What it does:
- For each language in LANG_MB with a target > 0 MB, it:
  - Streams split="train" from the "wiki40b" dataset
  - Shuffles the stream
  - Keeps examples where len(example["text"]) > 3000 characters
  - Writes JSONL lines to a file named:
      {lang}_{split}_{target}MB.jsonl
- It stops writing as soon as the output file size meets or exceeds the target size.

How to run:
    python download_wiki40b

2) clean_wiki40b.py
~~~~~~~~~~~~~
What it does:
- Looks for input files in directory INPUT_DIR
- For each input file it:
  - Parses each line as JSON (skips malformed lines)
  - Decodes bytes-literal strings like b"..." / b'...' in certain fields
  - Cleans the "text" field by:
    - Replacing Wiki40B marker tokens (e.g. _NEWLINE_, _START_ARTICLE_) with whitespace
    - Removing leftover marker-like ALLCAPS tokens
    - Cutting off trailing sections when a line matches headings like:
      "References", "External links", "See also", "Further reading" (English)
    - Collapsing whitespace and trimming
  - Drops records whose cleaned text is shorter than MIN_CHARS_AFTER_CLEAN (default 200)
  - Writes the cleaned records as a single JSON array to:
      OUTPUT_DIR = "train_clean"
      {input_basename}.json

How to run:
    python clean_wiki40b.py