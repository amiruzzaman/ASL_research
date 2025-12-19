import torch


class Vocabulary:
    def __init__(self, words: str):
        self.words = words

        # Create dictionaries to convert string tokens into their ids and vice versa
        self.word_to_idx = {word: i for i, word in enumerate(self.words)}
        self.idx_to_word = {i: word for i, word in enumerate(self.words)}

        assert "<pad>" in self.word_to_idx, "<PAD> token doesn't exist in text vocab"
        assert "<eos>" in self.word_to_idx, "<EOS> token doesn't exist in text vocab"
        assert "<sos>" in self.word_to_idx, "<SOS> token doesn't exist in text vocab"
        assert "<unk>" in self.word_to_idx, "<UNK> token doesn't exist in text vocab"

        self.sos_token = self.word_to_idx["<sos>"]
        self.eos_token = self.word_to_idx["<eos>"]
        self.pad_token = self.word_to_idx["<pad>"]
        self.unk_token = self.word_to_idx["<unk>"]

    def tokenize(self, sentence: str):
        return torch.tensor(
            [self.word_to_idx["<sos>"]]
            + [
                (self.word_to_idx[word] if word in self.word_to_idx else self.unk_token)
                for word in sentence.split()
            ]
            + [self.word_to_idx["<eos>"]]
        )

    def tokenize_batch(self, sentences: list):
        return torch.stack([self.tokenize(sentence) for sentence in sentences], dim=0)

    def decode(self, sentence: list):
        sentence = list(filter(self.remove_special_tokens, sentence))
        return " ".join([self.idx_to_word[token] for token in sentence])

    def decode_batch(self, sentences: list):
        return [self.decode(sentence) for sentence in sentences]

    def remove_special_tokens(self, token: int):
        return (
            token != self.pad_token
            and token != self.eos_token
            and token != self.sos_token
        )

    def get_size(self):
        return len(self.word_to_idx)
