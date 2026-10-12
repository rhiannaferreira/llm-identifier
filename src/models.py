"""
models.py — The two required architectures (CNN and RNN/LSTM).

Both take padded token-id sequences and return logits over the 5 generators.
They share an embedding interface so the training loop is architecture-agnostic.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from data import PAD_ID


class TextCNN(nn.Module):
    """Kim (2014)-style CNN: parallel n-gram filters + max-over-time pooling.

    Filter widths 2/3/4/5 give the model access to short stylistic n-grams
    ("In conclusion ,", "* *", "As an AI"), and max-pooling means a single
    strong giveaway anywhere in the document can drive the prediction.
    """

    def __init__(self, vocab_size, num_classes, embed_dim=128,
                 num_filters=100, filter_sizes=(2, 3, 4, 5), dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_ID)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, kernel_size=k) for k in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)
        self.min_len = max(filter_sizes)

    def forward(self, x, lengths=None):
        # Guard: a sequence shorter than the widest filter would break conv1d.
        if x.size(1) < self.min_len:
            pad = self.min_len - x.size(1)
            x = F.pad(x, (0, pad), value=PAD_ID)
        emb = self.embedding(x).transpose(1, 2)        # (B, E, T)
        feats = [F.relu(conv(emb)).max(dim=2).values for conv in self.convs]
        h = self.dropout(torch.cat(feats, dim=1))
        return self.fc(h)


class TextLSTM(nn.Module):
    """Bidirectional LSTM with masked mean+max pooling over the hidden states.

    Pooling over all timesteps rather than using only the final hidden state
    matters here: a fingerprint can sit anywhere in a long response, and the
    final state of a 256-token sequence tends to forget the opening.
    Packing ensures padding never contributes to the recurrence.
    """

    def __init__(self, vocab_size, num_classes, embed_dim=128,
                 hidden_dim=128, num_layers=1, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_ID)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 4, num_classes)   # mean+max of bi-dir

    def forward(self, x, lengths):
        emb = self.embedding(x)
        lengths_cpu = lengths.cpu().clamp(min=1)
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths_cpu, batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(
            out, batch_first=True, total_length=x.size(1))

        mask = (torch.arange(x.size(1), device=x.device)[None, :]
                < lengths[:, None]).unsqueeze(-1)
        out = out.masked_fill(~mask, 0.0)

        mean = out.sum(dim=1) / lengths.clamp(min=1).unsqueeze(-1).float()
        maxed = out.masked_fill(~mask, float("-inf")).max(dim=1).values
        maxed = torch.nan_to_num(maxed, neginf=0.0)

        h = self.dropout(torch.cat([mean, maxed], dim=1))
        return self.fc(h)


def build_model(name, vocab_size, num_classes, **kwargs):
    if name == "cnn":
        return TextCNN(vocab_size, num_classes, **kwargs)
    if name == "lstm":
        return TextLSTM(vocab_size, num_classes, **kwargs)
    raise ValueError(f"unknown model: {name}")
