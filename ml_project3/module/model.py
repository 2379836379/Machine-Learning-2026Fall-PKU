import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Seq2SeqGRU(nn.Module):
    """Bidirectional GRU encoder with an attention-based autoregressive decoder."""
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int,
                 embed_drop: float = 0.2, hidden_drop: float = 0.1,
                 pad_id: int = 0, bos_id: int = 2, eos_id: int = 3):
        super().__init__()
        if hidden_dim % 2:
            raise ValueError("hidden_dim must be even for the bidirectional encoder")
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.dropout = nn.Dropout(embed_drop)
        self.encoder = nn.GRU(embed_dim, hidden_dim // 2, batch_first=True,
                              bidirectional=True)
        self.init_decoder = nn.Linear(hidden_dim, hidden_dim)
        self.key_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.query_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.attention_score = nn.Linear(hidden_dim, 1, bias=False)
        self.decoder = nn.GRU(embed_dim + hidden_dim, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim * 2, vocab_size)
        self.hidden_dropout = nn.Dropout(hidden_drop)

    def encode(self, input_ids, source_mask):
        embedded = self.dropout(self.embedding(input_ids))
        lengths = (~source_mask).sum(dim=1).clamp(min=1).cpu()
        packed = pack_padded_sequence(embedded, lengths, batch_first=True,
                                      enforce_sorted=False)
        packed_memory, hidden = self.encoder(packed)
        memory, _ = pad_packed_sequence(packed_memory, batch_first=True,
                                        total_length=input_ids.size(1))
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=-1)
        decoder_hidden = torch.tanh(self.init_decoder(hidden)).unsqueeze(0)
        return memory, decoder_hidden

    def forward(self, input_ids, source_mask, decoder_input_ids):
        memory, hidden = self.encode(input_ids, source_mask)
        keys = self.key_projection(memory)
        logits = []
        for step in range(decoder_input_ids.size(1)):
            token = decoder_input_ids[:, step:step + 1]
            query = self.query_projection(hidden[-1]).unsqueeze(1)
            scores = self.attention_score(torch.tanh(keys + query)).squeeze(-1)
            scores = scores.masked_fill(source_mask, -1e4)
            weights = torch.softmax(scores, dim=-1).unsqueeze(1)
            context = torch.bmm(weights, memory)
            decoder_input = torch.cat((self.dropout(self.embedding(token)), context), dim=-1)
            output, hidden = self.decoder(decoder_input, hidden)
            output = self.hidden_dropout(output)
            logits.append(self.output(torch.cat((output, context), dim=-1)))
        return torch.cat(logits, dim=1)

    @torch.no_grad()
    def generate(self, input_ids, source_mask, max_length: int, bos_id=None, eos_id=None):
        bos_id = self.bos_id if bos_id is None else bos_id
        eos_id = self.eos_id if eos_id is None else eos_id
        memory, hidden = self.encode(input_ids, source_mask)
        keys = self.key_projection(memory)
        batch_size = input_ids.size(0)
        token = input_ids.new_full((batch_size, 1), bos_id)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)
        generated = []
        for _ in range(max_length):
            query = self.query_projection(hidden[-1]).unsqueeze(1)
            scores = self.attention_score(torch.tanh(keys + query)).squeeze(-1)
            scores = scores.masked_fill(source_mask, -1e4)
            context = torch.bmm(torch.softmax(scores, dim=-1).unsqueeze(1), memory)
            decoder_input = torch.cat((self.embedding(token), context), dim=-1)
            output, hidden = self.decoder(decoder_input, hidden)
            token = self.output(torch.cat((output, context), dim=-1)).argmax(dim=-1)
            token = torch.where(finished.unsqueeze(1), token.new_full(token.shape, self.pad_id), token)
            generated.append(token)
            finished |= token.squeeze(1).eq(eos_id)
            if finished.all():
                break
        if not generated:
            return input_ids.new_empty((batch_size, 0))
        return torch.cat(generated, dim=1)


class Transformer(nn.Module):
    def __init__(self, vocab_size: int, max_seq_len: int, embed_dim: int, hidden_dim: int, n_layer: int, n_head: int, ff_dim: int, embed_drop: float, hidden_drop: float):
        super().__init__()
        self.tok_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(max_seq_len, embed_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_head, dim_feedforward=ff_dim, dropout=hidden_drop)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layer)
        self.embed_dropout = nn.Dropout(embed_drop)
        self.linear1 = nn.Linear(embed_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, embed_dim)

    def encode(self, x, mask):
        x = x.transpose(0, 1)
        x = self.encoder(x, src_key_padding_mask=mask)
        x = x.transpose(0, 1)
        return x

    def forward(self, x, *args):
        # (batch_size, max_seq_len, embed_dim)
        mask = args[0] if len(args) > 0 else None
        tok_emb = self.tok_embedding(x)
        max_seq_len = x.shape[-1]
        pos_emb = self.pos_embedding(torch.arange(max_seq_len).to(x.device))
        x = tok_emb + pos_emb.unsqueeze(0)
        x = self.embed_dropout(x)
        x = self.linear1(x)
        x = self.encode(x, mask)
        x = self.linear2(x)
        probs = torch.matmul(x, self.tok_embedding.weight.t())
        return probs


class BiLSTM(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, n_layer: int, embed_drop: float, rnn_drop: float):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.bilstm = nn.LSTM(embed_dim, hidden_dim // 2, num_layers=n_layer,
                              dropout=rnn_drop if n_layer > 1 else 0, batch_first=True, bidirectional=True)
        self.embed_dropout = nn.Dropout(embed_drop)
        self.linear = nn.Linear(hidden_dim, embed_dim)

    def encode(self, x):
        x = self.embedding(x)
        x = self.embed_dropout(x)
        x, _ = self.bilstm(x)
        return x

    def predict(self, x):
        x = self.linear(x)
        probs = torch.matmul(x, self.embedding.weight.t())
        return probs

    def forward(self, x, *args):
        x = self.encode(x)
        return self.predict(x)


class BiLSTMAttn(BiLSTM):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, n_layer: int, embed_drop: float, rnn_drop: float, n_head: int):
        super().__init__(vocab_size, embed_dim, hidden_dim, n_layer, embed_drop, rnn_drop)
        self.attn = nn.MultiheadAttention(hidden_dim, n_head)

    def forward(self, x, *args):
        mask = args[0] if len(args) > 0 else None
        x = self.encode(x)
        x = x.transpose(0, 1)
        x = self.attn(x, x, x, key_padding_mask=mask)[0].transpose(0, 1)
        return self.predict(x)


class BiLSTMCNN(BiLSTM):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, n_layer: int, embed_drop: float, rnn_drop: float):
        super().__init__(vocab_size, embed_dim, hidden_dim, n_layer, embed_drop, rnn_drop)
        self.conv = nn.Conv1d(in_channels=hidden_dim,
                              out_channels=hidden_dim, kernel_size=3, padding=1)

    def forward(self, x, *args):
        x = self.encode(x)
        x = x.transpose(1, 2)
        x = self.conv(x).transpose(1, 2).relu()
        return self.predict(x)


class BiLSTMConvAttRes(BiLSTM):
    def __init__(self, vocab_size: int, max_seq_len: int, embed_dim: int, hidden_dim: int, n_layer: int, embed_drop: float, rnn_drop: float, n_head: int):
        super().__init__(vocab_size, embed_dim, hidden_dim, n_layer, embed_drop, rnn_drop)
        self.attn = nn.MultiheadAttention(hidden_dim, n_head)
        self.conv = nn.Conv1d(in_channels=hidden_dim,
                              out_channels=hidden_dim, kernel_size=3, padding=1)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, *args):
        mask = args[0] if len(args) > 0 else None
        x = self.encode(x)
        res = x
        x = self.conv(x.transpose(1, 2)).relu()
        x = x.permute(2, 0, 1)
        x = self.attn(x, x, x, key_padding_mask=mask)[0].transpose(0, 1)
        x = self.norm(res + x)
        return self.predict(x)


class CNN(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, embed_drop: float):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.conv = nn.Conv1d(in_channels=embed_dim,
                              out_channels=hidden_dim, kernel_size=3, padding=1)
        self.embed_dropout = nn.Dropout(embed_drop)
        self.linear = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x, *args):
        x = self.embedding(x)
        x = self.embed_dropout(x)
        x = x.transpose(1, 2)
        x = self.conv(x).transpose(1, 2).relu()
        x = self.linear(x)
        probs = torch.matmul(x, self.embedding.weight.t())
        return probs
