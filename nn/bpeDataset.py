import torch
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer
import json
import os
from typing import Optional, Tuple, List

class BPETokenizedDataset(Dataset):
    """
    Dataset for language modeling using pre-tokenized BPE corpus.
    The corpus has space-separated tokens, with <sp> representing whitespace.
    """
    def __init__(self, tokenized_corpus_path: str, seq_length: int, tokenizer_path: str = "bpe_tokenizer.json"):
        """
        Args:
            tokenized_corpus_path: path to ngram_training_corpus.txt
            seq_length: c (context length in tokens)
            tokenizer_path: path to the saved tokenizer JSON file
        """
        self.seq_length = seq_length
        self.tokenized_corpus_path = tokenized_corpus_path
        
        # Load the tokenizer to get vocab mappings
        print(f"Loading tokenizer from {tokenizer_path}...")
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        
        # Build token-to-id mapping
        self.token_to_id = self.tokenizer.get_vocab()
        
        # Read and tokenize the corpus
        print(f"Loading tokenized corpus from {tokenized_corpus_path}...")
        all_tokens = []
        doc_count = 0
        
        with open(tokenized_corpus_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                # Split line into tokens (they're already space-separated)
                tokens = line.strip().split()
                
                # Convert tokens to IDs
                for token in tokens:
                    if token in self.token_to_id:
                        all_tokens.append(self.token_to_id[token])
                    else:
                        # Handle unknown tokens with [UNK]
                        unk_id = self.token_to_id.get("[UNK]", 0)
                        all_tokens.append(unk_id)
                
                # Add [EOS] between documents
                eos_id = self.token_to_id.get("[EOS]", 1)
                all_tokens.append(eos_id)
                
                doc_count += 1
                
                if (line_num + 1) % 10000 == 0:
                    print(f"  Processed {line_num + 1} documents, {len(all_tokens):,} tokens")
        
        # Convert to tensor
        self.tokens = torch.tensor(all_tokens, dtype=torch.long)
        
        print(f"\nTotal tokens in corpus: {len(self.tokens):,}")
        print(f"Vocabulary size: {len(self.token_to_id)}")
        print(f"Number of documents: {doc_count}")
        print(f"Total sequences available: {len(self):,}")
        
    def __len__(self):
        # Number of possible sequences
        return max(0, len(self.tokens) - self.seq_length - 1)
    
    def __getitem__(self, idx):
        """
        Returns a sequence of token IDs of length seq_length+1
        """
        return self.tokens[idx:idx + self.seq_length + 1]


class StreamingBPEDataset(Dataset):
    """
    Memory-efficient version that reads from the tokenized corpus on-the-fly.
    Useful for very large corpora that don't fit in memory.
    """
    def __init__(self, tokenized_corpus_path: str, seq_length: int, 
                 tokenizer_path: str = "bpe_tokenizer.json", 
                 max_samples: Optional[int] = None):
        self.seq_length = seq_length
        self.tokenized_corpus_path = tokenized_corpus_path
        self.max_samples = max_samples
        
        # Load tokenizer
        print(f"Loading tokenizer from {tokenizer_path}...")
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.token_to_id = self.tokenizer.get_vocab()
        
        # Build index of valid starting positions
        print("Building index of valid sequence starts...")
        self.starts = []  # List of (doc_idx, token_pos)
        doc_idx = 0
        
        with open(tokenized_corpus_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                tokens = line.strip().split()
                token_ids = []
                
                # Convert tokens to IDs
                for token in tokens:
                    if token in self.token_to_id:
                        token_ids.append(self.token_to_id[token])
                    else:
                        token_ids.append(self.token_to_id.get("[UNK]", 0))
                
                # Add EOS token
                doc_tokens = token_ids + [self.token_to_id.get("[EOS]", 1)]
                
                # Add all valid sequence starts from this document
                for i in range(len(doc_tokens) - seq_length - 1):
                    self.starts.append((doc_idx, i))
                    
                    if max_samples and len(self.starts) >= max_samples:
                        break
                
                doc_idx += 1
                
                if max_samples and len(self.starts) >= max_samples:
                    break
                
                if (line_num + 1) % 10000 == 0:
                    print(f"  Indexed {line_num + 1} documents, {len(self.starts):,} sequences")
        
        print(f"Total sequences available: {len(self.starts):,}")
    
    def __len__(self):
        return len(self.starts)
    
    def __getitem__(self, idx):
        doc_idx, pos = self.starts[idx]
        
        # Find the right document and extract the sequence
        with open(self.tokenized_corpus_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i == doc_idx:
                    tokens = line.strip().split()
                    token_ids = []
                    
                    for token in tokens:
                        if token in self.token_to_id:
                            token_ids.append(self.token_to_id[token])
                        else:
                            token_ids.append(self.token_to_id.get("[UNK]", 0))
                    
                    # Add EOS
                    token_ids.append(self.token_to_id.get("[EOS]", 1))
                    
                    # Extract sequence
                    seq = token_ids[pos:pos + self.seq_length + 1]
                    
                    # Pad if necessary (shouldn't happen with proper indexing)
                    if len(seq) < self.seq_length + 1:
                        pad_id = self.token_to_id.get("[PAD]", 2)
                        seq.extend([pad_id] * (self.seq_length + 1 - len(seq)))
                    
                    return torch.tensor(seq, dtype=torch.long)
            
            # Fallback (should never reach here)
            pad_id = self.token_to_id.get("[PAD]", 2)
            return torch.full((self.seq_length + 1,), pad_id, dtype=torch.long)


def collate_batch(batch):
    """Stack sequences into a batch."""
    return torch.stack(batch)


def create_dataloaders_from_corpus(
    tokenized_corpus_path: str,
    seq_length: int,
    batch_size: int,
    tokenizer_path: str = "bpe_tokenizer.json",
    train_split: float = 0.9,
    max_samples: Optional[int] = None,
    use_streaming: bool = False,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, Tokenizer, int]:
    """
    Create dataloaders directly from the tokenized corpus.
    
    Args:
        tokenized_corpus_path: path to ngram_training_corpus.txt
        seq_length: context length in tokens
        batch_size: batch size
        tokenizer_path: path to the saved tokenizer JSON
        train_split: proportion of data for training
        max_samples: limit total samples (for debugging)
        use_streaming: use memory-efficient streaming dataset
        num_workers: number of dataloader workers
    
    Returns:
        train_loader, val_loader, tokenizer, vocab_size
    """
    # Create dataset
    if use_streaming:
        dataset = StreamingBPEDataset(
            tokenized_corpus_path=tokenized_corpus_path,
            seq_length=seq_length,
            tokenizer_path=tokenizer_path,
            max_samples=max_samples
        )
    else:
        dataset = BPETokenizedDataset(
            tokenized_corpus_path=tokenized_corpus_path,
            seq_length=seq_length,
            tokenizer_path=tokenizer_path
        )
    
    # Split into train/val
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    
    if max_samples and not use_streaming:
        train_size = min(train_size, max_samples)
        val_size = min(val_size, max_samples // 10)
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_batch,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_batch,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Get vocab size from tokenizer
    tokenizer = Tokenizer.from_file(tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()
    
    return train_loader, val_loader, tokenizer, vocab_size