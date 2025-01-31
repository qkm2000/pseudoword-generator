import torch
import string


class wordTokenizer:
    def __init__(self, max_length=12):
        # Define vocabulary: lowercase letters + special tokens
        self.vocab = ['<pad>', '<start>', '<end>'] + list(string.ascii_lowercase)
        self.token2idx = {token: idx for idx, token in enumerate(self.vocab)}
        self.idx2token = {idx: token for token, idx in self.token2idx.items()}

        self.pad_token_id = self.token2idx['<pad>']
        self.start_token_id = self.token2idx['<start>']
        self.end_token_id = self.token2idx['<end>']
        self.max_length = max_length

    def encode(self, word):
        # Convert word to token indices

        tokens = ['<start>'] + list(word.lower()) + ['<end>']
        ids = [self.token2idx[t] for t in tokens]

        # Pad sequence
        if len(ids) < self.max_length:
            ids = ids + [self.pad_token_id] * (self.max_length - len(ids))
        else:
            ids = ids[:self.max_length]

        return torch.tensor(ids)

    def collate_fn(self, batch):
        # Unpack numbers and words from batch
        numbers, words = zip(*batch)

        # Convert to tensors
        number_tensor = torch.tensor(numbers, dtype=torch.float)
        word_tensors = [self.encode(word) for word in words]
        word_tensor = torch.stack(word_tensors)

        return number_tensor, word_tensor

    def decode(self, indices):
        # Convert token indices back to word
        tokens = [self.idx2token[idx.item()] for idx in indices]
        # Remove special tokens and join
        word = ''.join([t for t in tokens if t not in ['<pad>', '<start>', '<end>']])
        return word
