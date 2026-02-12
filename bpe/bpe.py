from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import json

# When run, this takes a single JSON file containing the corpus and creates a vocab in vocab.json
# outputs ngram_training_corpus.txt which is the tokenized version of the training corpus

# Make sure to run this code whilst in the /EMZ-cse447-project/bpe directory, as the files written will be written in your current directory
# Also it won't be able to find the proper json data file

## TODO: make it read and write from the same place consistently, irregardless of the user's directory

# File path for single JSON file.
json_path_str = '../data/wiki40b/train_small/en_train_10MB.json'

# Vocab size (8000 was default)
vocab_size_parameter = 8000

# Initializes tokenizer
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = Whitespace()

trainer = BpeTrainer(
    vocab_size=vocab_size_parameter,
    special_tokens=["[UNK]", "[PAD]", "[BOS]", "[EOS]"],
    min_frequency=2  # Ignore tokens that appear less than twice
)

# Load texts from JSON
def text_generator(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for item in data:
            yield item['text']


# Convert generator to list for training
texts = list(text_generator(json_path_str))

# Trains the BPE
tokenizer.train_from_iterator(texts, trainer=trainer)

# Save vocabulary as JSON in ./vocab.json
tokenizer.model.save('.')

# Save the full tokenizer
tokenizer.save("bpe_tokenizer.json")

# Creates a json file of the tokenized corpus for N-gram training
# Found at ./ngram_training_corpus.txt
with open('ngram_training_corpus.txt', 'w', encoding='utf-8') as f:
    for text in texts:
        # Encode and join with spaces
        encoded = tokenizer.encode(text)
        f.write(" ".join(encoded.tokens) + "\n")

print("Tokenizer and corpus ready!")