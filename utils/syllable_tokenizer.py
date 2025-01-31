from transformers import PreTrainedTokenizer
from typing import List, Optional, Dict
import json
import re
import os


class SyllableTokenizer(PreTrainedTokenizer):
    def __init__(
        self,
        vocab_file=r"utils/tokenizer-vocab.json",
        unk_token="<unk>",
        pad_token="<pad>",
        eos_token="</s>",
        bos_token="<s>",
        **kwargs
    ):

        # Initialize vocabulary
        self.load_vocabulary(vocab_file)
        self.vocab.update(
            {
                pad_token: 0,
                eos_token: 1,
                bos_token: 2,
                unk_token: 3,
            }
        )
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        self.next_token_id = len(self.vocab)

        super().__init__(
            pad_token=pad_token,
            eos_token=eos_token,
            bos_token=bos_token,
            unk_token=unk_token,
            **kwargs
        )

    def split_word(self, word: str) -> List[str]:
        """Split a word into alphabets"""
        return [char for char in word]

    def _tokenize(self, text: str) -> List[str]:
        """Split text into syllables."""
        # Split into words first
        text = re.sub(r'[^\w\s]', '', text)
        text = text.lower()
        words = text.split()
        syllables = []

        for word in words:
            # Find all syllables in the word
            word_syllables = self.split_word(word)
            if not word_syllables:
                # If no syllables found, treat the whole word as one token
                syllables.append(word)
            else:
                syllables.extend(word_syllables)

        return syllables

    def _convert_token_to_id(self, token: str) -> int:
        """Convert a token to its ID, adding to vocabulary if necessary."""
        if token not in self.vocab:
            # Add new token to vocabulary
            self.vocab[token] = self.next_token_id
            self.id_to_token[self.next_token_id] = token
            self.next_token_id += 1
        return self.vocab[token]

    def _convert_id_to_token(self, index: int) -> str:
        """Convert an ID back to its token."""
        return self.id_to_token.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """Convert a sequence of tokens to a single string."""
        return "".join(tokens)

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> tuple:
        """Save the vocabulary to a file."""
        if not os.path.isdir(save_directory):
            os.makedirs(save_directory)

        vocab_file = os.path.join(
            save_directory,
            (filename_prefix + "-" if filename_prefix else "") + "vocab.json"
        )

        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)

        return (vocab_file,)

    def load_vocabulary(self, vocab_file: str):
        """Load vocabulary from a file."""
        import json

        with open(vocab_file, "r", encoding="utf-8") as f:
            self.vocab = json.load(f)

        self.id_to_token = {v: k for k, v in self.vocab.items()}
        self.next_token_id = max(self.vocab.values()) + 1

    @property
    def vocab_size(self) -> int:
        """Return the size of vocabulary."""
        return len(self.vocab)

    def get_vocab(self) -> Dict[str, int]:
        """Return the vocabulary dictionary."""
        return self.vocab.copy()


if __name__ == "__main__":
    # Example usage
    tokenizer = SyllableTokenizer(vocab_file="datasets/alphabets.json")
    text = "tokyo"
    tokens = tokenizer.encode(text)
    print(tokens)
    print(tokenizer.decode(tokens))
