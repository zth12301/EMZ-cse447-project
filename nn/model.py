import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from bpeDataset import create_dataloaders_from_corpus
from itertools import cycle

class TransformerModel(nn.Module):
    """
    Single layer, single attention head, + 1 MLP block.
    Columns = token positions in the context.

        E   : (d_emb, c)
        Q,K : (d_qk, c)
        S   : (c, c) = Q^T K
        A   : (c, c) = softmax((S + M)/sqrt(d_qk)) row-wise
        V   : (d_emb, c) = W_vup (W_vdown E)
        O   : (d_emb, c) = V A^T
        AttnOut: (d_emb, c) = W_o O
        Residual: E + AttnOut

    Pre-norm is used:
        E1 = E + Attn(LN(E))
        E2 = E1 + MLP(LN(E1))
    """

    def __init__(self, vocab_size: int, d_emb: int, d_qk: int, d_ff: int, causal: bool = True, tie_weights: bool = True):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_emb = d_emb
        self.d_qk = d_qk
        self.d_ff = d_ff
        self.causal = causal

        # Embedding matrix We: (d_emb, vocab_size)
        # Token ids -> E by selecting columns of We
        self.We = nn.Parameter(torch.randn(d_emb, vocab_size) * 0.02)

        # LayerNorms over the embedding dimension (applied per position)
        self.ln1 = nn.LayerNorm(d_emb)
        self.ln2 = nn.LayerNorm(d_emb)

        # Attention matrices
        # Wq, Wk: (d_qk, d_emb)
        self.Wq = nn.Parameter(torch.randn(d_qk, d_emb) * 0.02)
        self.Wk = nn.Parameter(torch.randn(d_qk, d_emb) * 0.02)

        # Low-rank value projection
        # W_vdown: (d_qk, d_emb), W_vup: (d_emb, d_qk)
        self.W_vdown = nn.Parameter(torch.randn(d_qk, d_emb) * 0.02)
        self.W_vup   = nn.Parameter(torch.randn(d_emb, d_qk) * 0.02)

        # Output projection Wo: (d_emb, d_emb)
        self.Wo = nn.Parameter(torch.randn(d_emb, d_emb) * 0.02)

        # MLP matrices (position-wise)
        # W1: (d_ff, d_emb), b1: (d_ff, 1)
        # W2: (d_emb, d_ff), b2: (d_emb, 1)
        self.W1 = nn.Parameter(torch.randn(d_ff, d_emb) * 0.02)
        self.b1 = nn.Parameter(torch.zeros(d_ff, 1))
        self.W2 = nn.Parameter(torch.randn(d_emb, d_ff) * 0.02)
        self.b2 = nn.Parameter(torch.zeros(d_emb, 1))

        # W_out: (vocab_size, d_emb), b_out: (vocab_size, 1)
        self.W_out = nn.Parameter(torch.randn(vocab_size, d_emb) * 0.02)
        self.b_out = nn.Parameter(torch.zeros(vocab_size, 1))

        # Tie output weights to embedding weights (weight tying)
        # We is (d_emb, vocab). W_out should be (vocab, d_emb) = We^T.
        self.tie_weights = tie_weights

    def _embed(self, tokens: torch.LongTensor) -> torch.Tensor:
        """
        tokens: (B, c) ints
        returns E: (B, d_emb, c) with columns = positions
        """
        # We[:, tokens] would yield (d_emb, B, c); permute to (B, d_emb, c)
        E = self.We[:, tokens].permute(1, 0, 2).contiguous()
        return E

    def _apply_layernorm_columns(self, E: torch.Tensor, ln: nn.LayerNorm) -> torch.Tensor:
        """
        E: (B, d_emb, c) -> apply LN over d_emb for each position
        nn.LayerNorm expects last dim = normalized features, so we transpose to (B, c, d_emb).
        """
        x = E.transpose(1, 2)          # (B, c, d_emb)
        x = ln(x)                      # (B, c, d_emb)
        return x.transpose(1, 2)       # (B, d_emb, c)

    def _causal_mask(self, c: int, device) -> torch.Tensor:
        """
        Returns M: (c, c) with -inf where j > i (future positions disallowed), else 0.
        """
        # upper triangular (excluding diagonal) -> masked
        mask = torch.triu(torch.ones(c, c, device=device, dtype=torch.bool), diagonal=1)
        M = torch.zeros(c, c, device=device, dtype=torch.float32)
        M = M.masked_fill(mask, float("-inf"))
        return M

    def attention(self, E: torch.Tensor) -> torch.Tensor:
        """
        E: (B, d_emb, c)
        Returns AttnOut: (B, d_emb, c)
        """
        B, d_emb, c = E.shape
        assert d_emb == self.d_emb

        # Q = Wq E, K = Wk E  -> (B, d_qk, c)
        Q = torch.matmul(self.Wq, E)   # (d_qk, d_emb) @ (B, d_emb, c) => broadcast -> (B, d_qk, c)
        K = torch.matmul(self.Wk, E)

        # S = Q^T K -> (B, c, c)
        # Q^T: (B, c, d_qk), K: (B, d_qk, c)
        S = torch.matmul(Q.transpose(1, 2), K)  # (B, c, c)

        # Scale for stability
        S = S / math.sqrt(self.d_qk)

        # Add causal mask
        if self.causal:
            M = self._causal_mask(c, device=E.device)     # (c, c)
            S = S + M.unsqueeze(0)                        # (B, c, c)

        # A = softmax(S) row-wise over keys j (dim=-1)
        A = F.softmax(S, dim=-1)                          # (B, c, c)

        # V = W_vup (W_vdown E) -> (B, d_emb, c)
        V_down = torch.matmul(self.W_vdown, E)            # (B, d_qk, c)
        V = torch.matmul(self.W_vup, V_down)              # (B, d_emb, c)

        # O = V A^T -> (B, d_emb, c)
        O = torch.matmul(V, A.transpose(1, 2))            # (B, d_emb, c)

        # AttnOut = Wo O -> (B, d_emb, c)
        AttnOut = torch.matmul(self.Wo, O)                # (B, d_emb, c)
        return AttnOut

    def mlp(self, E: torch.Tensor) -> torch.Tensor:
        """
        E: (B, d_emb, c)
        Returns: (B, d_emb, c)
        Position-wise MLP matching:
          H = W1 E + b1, U = GELU(H), out = W2 U + b2
        """
        # H: (B, d_ff, c)
        H = torch.matmul(self.W1, E) + self.b1            # b1 broadcasts along batch and positions
        U = F.gelu(H)
        out = torch.matmul(self.W2, U) + self.b2          # (B, d_emb, c)
        return out

    def logits_from_E(self, E: torch.Tensor) -> torch.Tensor:
        """
        E: (B, d_emb, c)
        Returns Logits: (B, vocab_size, c)
        """
        if self.tie_weights:
            # W_out = We^T (vocab, d_emb)
            W_out = self.We.transpose(0, 1)  # (vocab, d_emb)
        else:
            W_out = self.W_out

        # (vocab, d_emb) @ (B, d_emb, c) => (B, vocab, c)
        Logits = torch.matmul(W_out, E) + self.b_out
        return Logits

    def forward(self, tokens: torch.LongTensor) -> torch.Tensor:
        """
        tokens: (B, c)
        Returns Logits: (B, vocab, c) (columns = positions)
        """
        # E: (B, d_emb, c)
        E = self._embed(tokens)

        # Attention block: E1 = E + Attn(LN(E))
        E_ln = self._apply_layernorm_columns(E, self.ln1)
        E1 = E + self.attention(E_ln)

        # MLP block: E2 = E1 + MLP(LN(E1))
        E1_ln = self._apply_layernorm_columns(E1, self.ln2)
        E2 = E1 + self.mlp(E1_ln)

        Logits = self.logits_from_E(E2)  # (B, vocab, c)
        return Logits
    
    def loss(self, tokens: torch.LongTensor) -> torch.Tensor:
        """
        Next-token (teacher forcing) loss.
        tokens: (B, c+1) so we can predict every next token for c steps.

        Uses:
          input  = tokens[:, :-1]  shape (B, c)
          target = tokens[:,  1:]  shape (B, c)

        Logits returned are (B, vocab, c) aligned with target positions 1..c.
        """
        # Ground truth input tokens
        x = tokens[:, :-1]          # (B, c)
        y = tokens[:, 1:]           # (B, c)

        logits = self.forward(x)    # (B, vocab, c)

        # CrossEntropyLoss expects (N, C) logits and (N,) targets.
        B, V, c = logits.shape
        logits_2d = logits.permute(0, 2, 1).reshape(B * c, V)  # (B*c, vocab)
        y_1d = y.reshape(B * c)                                # (B*c,)

        return F.cross_entropy(logits_2d, y_1d)


_train_loader_iterator = None
_val_loader_iterator = None
_current_epoch = 0


def init_data_loaders(tokenized_corpus_path: str, seq_length: int, batch_size: int, 
                      tokenizer_path: str = "bpe_tokenizer.json"):
    """
    Initialize the dataloaders and create iterators.
    Call this once before training.
    """
    global _train_loader_iterator, _val_loader_iterator
    
    print("Creating dataloaders from BPE corpus...")
    train_loader, val_loader, tokenizer, vocab_size = create_dataloaders_from_corpus(
        tokenized_corpus_path=tokenized_corpus_path,
        seq_length=seq_length,
        batch_size=batch_size,
        tokenizer_path=tokenizer_path,
        train_split=0.9,
        use_streaming=False,
        num_workers=4
    )
    
    # Create infinite iterators
    _train_loader_iterator = cycle(train_loader)
    _val_loader_iterator = cycle(val_loader)
    
    print(f"Vocabulary size from BPE: {vocab_size}")
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")
    
    return vocab_size, len(train_loader), len(val_loader)


def next_batch(*, batch_size: int, ctxt_plus_one: int, device: str, 
               validation: bool = False) -> torch.LongTensor:
    """
    Returns a batch of token IDs for next-token training using BPE data.
    
    Inputs:
      batch_size: B
      ctxt_plus_one: c+1 (so the model can predict c next tokens)
      device: "cpu" or "cuda"
      validation: if True, get batch from validation set
      
    Output:
      tokens: LongTensor of shape (B, c+1) with values in [0, vocab_size)
    """
    global _train_loader_iterator, _val_loader_iterator, _current_epoch
    
    # Select the appropriate iterator
    iterator = _val_loader_iterator if validation else _train_loader_iterator
    
    # Get the next batch
    batch = next(iterator)
    
    # Ensure batch is on the correct device and has the right shape
    batch = batch.to(device)
    
    # The batch from dataloader already has shape (batch_size, seq_length+1)
    # which matches what we need (ctxt_plus_one should equal seq_length+1)
    assert batch.shape[1] == ctxt_plus_one, \
        f"Expected sequence length {ctxt_plus_one}, got {batch.shape[1]}"
    
    return batch


def reset_epoch():
    """Call this at the start of each new epoch to reset validation iterator position."""
    global _current_epoch
    _current_epoch += 1
    print(f"Starting epoch {_current_epoch}")


# Main / train loop
if __name__ == "__main__":

    # Path to your pre-tokenized corpus
    tokenized_corpus_path = "../bpe/ngram_training_corpus.txt"
    tokenizer_path = "../bpe/bpe_tokenizer.json"
    
    # Model hyperparameters - adjusted for BPE
    seq_length = 128
    batch_size = 32

    
    vocab_size, num_train_batches, num_val_batches = init_data_loaders(
        tokenized_corpus_path=tokenized_corpus_path,
        seq_length=seq_length,
        batch_size=batch_size,
        tokenizer_path=tokenizer_path
    )

    # Initialize the model
    # vocab_size = 128
    d_emb = 32
    d_qk = 16
    d_ff = 64
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TransformerModel(vocab_size, d_emb, d_qk, d_ff, causal=True, tie_weights=True).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

    # Train loop
    model.train()
    for epoch in range(10):  # Multiple epochs
        reset_epoch()
        
        for step in range(num_train_batches):
            # Get batch using next_batch (matches original interface)
            batch = next_batch(
                batch_size=batch_size,
                ctxt_plus_one=seq_length + 1,
                device=device,
                validation=False
            )
            
            loss = model.loss(batch)
            
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            
            if step % 50 == 0:
                print(f"Epoch {epoch}, Step {step}, Loss: {float(loss):.4f}")
        
        # Validation (Commented out for now)
        # model.eval()
        # val_loss = 0
        # with torch.no_grad():
        #     for _ in range(min(num_val_batches, 100)):  # Limit validation batches
        #         val_batch = next_batch(
        #             batch_size=batch_size,
        #             ctxt_plus_one=seq_length + 1,
        #             device=device,
        #             validation=True
        #         )
        #         val_loss += model.loss(val_batch).item()
        # val_loss /= min(num_val_batches, 100)
        # print(f"Epoch {epoch}, Validation Loss: {val_loss:.4f}")
        # model.train()


    # Save at end of training
    save_path = "transformer_model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vocab_size": vocab_size,
            "d_emb": d_emb,
            "d_qk": d_qk,
            "d_ff": d_ff,
        },
        save_path,
    )
    print(f"Saved model to {save_path}")