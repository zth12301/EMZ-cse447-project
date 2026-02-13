from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Split
from tokenizers import Regex
import json
import glob

# Explicit list for flie directories
# json_paths = [
#     '../data/wiki40b/train_small/en_train_10MB.json',
#     '../data/wiki40b/train_small/en_train_20MB.json',
# ]

# Match multiple files
# json_paths = glob.glob('../data/wiki40b/train_small/*.json')

# Recursively find all JSON files in a directory
json_paths = glob.glob('../data/wiki40b/**/*.json', recursive=True)

# Vocab size (8000 was default)
# TODO: 20,000 chosen somewhat arbitrarily - May need to test for best parameter
vocab_size_parameter = 20000

# Initializes tokenizer - don't set pre_tokenizer yet
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

# Use Split pre-tokenizer to split on every character including whitespace
# This treats spaces as regular characters that can be merged
tokenizer.pre_tokenizer = Split(Regex(r'\s+|\w+'), behavior="isolated")

trainer = BpeTrainer(
    vocab_size=vocab_size_parameter,
    special_tokens=["[UNK]", "[PAD]", "[BOS]", "[EOS]", "<sp>"],  # Add <sp> as a special token
    min_frequency=2  # Ignore tokens that appear less than twice
)

# Load texts from multiple JSON files
def text_generator_multiple(json_paths):
    for json_path in json_paths:
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                print(f"Loaded {len(data)} documents from {json_path}")
                for item in data:
                    yield item['text']
        except FileNotFoundError:
            print(f"Warning: File {json_path} not found, skipping...")
        except json.JSONDecodeError:
            print(f"Warning: File {json_path} is not valid JSON, skipping...")
        except Exception as e:
            print(f"Warning: Error reading {json_path}: {e}, skipping...")


# Convert generator to list for training
print("Loading texts from JSON files...")
texts = list(text_generator_multiple(json_paths))
print(f"Total documents loaded: {len(texts)}")

# Preprocess texts to ensure whitespace is preserved
def preprocess_texts(texts):
    processed_texts = []
    for text in texts:
        # This ensures that spaces are treated as regular characters
        # by making them explicitly visible as whitespace characters
        processed_texts.append(text)
    return processed_texts

processed_texts = preprocess_texts(texts)

# Trains the BPE on the processed texts
print(f"Training BPE tokenizer with vocab size {vocab_size_parameter}...")
tokenizer.train_from_iterator(processed_texts, trainer=trainer)
print("Training complete!")

# Save vocabulary as JSON in ./vocab.json
tokenizer.model.save('.')
print("Vocabulary saved to ./vocab.json")

# Save the full tokenizer
tokenizer.save("bpe_tokenizer.json")
print("Tokenizer saved to bpe_tokenizer.json")

# Creates a text file of the tokenized corpus for N-gram training
print("Tokenizing corpus and saving to ngram_training_corpus.txt...")
with open('ngram_training_corpus.txt', 'w', encoding='utf-8') as f:
    for i, text in enumerate(processed_texts):
        # Encode the text
        encoded = tokenizer.encode(text)
        
        # Get the tokens
        tokens = encoded.tokens
        
        # Replace actual whitespace tokens with <sp>
        processed_tokens = ['<sp>' if token.strip() == '' else token for token in tokens]
        
        # Join with spaces for output
        f.write(" ".join(processed_tokens) + "\n")
        
        # Print progress every 1000 documents
        if (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1}/{len(processed_texts)} documents")

print(f"Tokenizer and corpus ready! Processed {len(processed_texts)} documents total.")

# Save metadata about the training run
metadata = {
    "vocab_size": vocab_size_parameter,
    "num_documents": len(processed_texts),
    "source_files": json_paths,
    "pre_tokenizer": "Split(Regex(r'\\s+|\\w+'), behavior='isolated')",
    "special_tokens": ["[UNK]", "[PAD]", "[BOS]", "[EOS]", "<sp>"],
    "min_frequency": 2
}

with open('bpe_training_metadata.json', 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=2)
print("Training metadata saved to bpe_training_metadata.json")