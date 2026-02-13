When run, bpe.py reads JSON files from data/wiki40b/, and byte pair encodes a vocabulary

Currently, it will read and write from the user's directory. It is intended to be ran from ./bpe
If not, it will write files elsewhere and likely fail to find the JSON files.

Outputs:
 - bpe_tokenizer.json: Data of the actual tokenizer, including original vocab set, merges, and metadata

 - bpe_training_metadata.json: Metadata of the data and tokenizer parameters

 - merges.txt: text file of the merges the byte pair encoder made

 + ngram_training_corpus.txt: text file of the tokenized training data for the byte pair encoder
   - Tokens seperated by whitespace, represents whitespace tokens as <s>
   - Articles/text entires seperated by newline

 - vocab.json: Vocab set (all the individual characters found in the training data)

The current files are the result of byte pair encoding all the data from the json files in /data/wiki40b